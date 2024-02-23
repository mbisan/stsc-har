import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

from data.base import STSDataset

from transforms.gaf_mtf import mtf_compute, gaf_compute
import pywt

class StreamingTimeSeriesCopy(Dataset):

    def __init__(self,
            stsds: STSDataset, indices: np.ndarray, label_mode: int = 1, mode: str = None, mtf_bins: int = 30, clr_indices: list[np.ndarray] = None
            ) -> None:
        super().__init__()

        self.stsds = stsds
        self.indices = indices
        self.label_mode = label_mode
        self.mode = mode
        self.mtf_bins = mtf_bins
        self.clr_indices = clr_indices
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:
        ts, c = self.stsds[self.indices[index]]
    
        if self.mode == "seg":
            return {"series": ts, "scs": c}

        if self.label_mode > 1:
            c_ = torch.mode(c[-self.label_mode:]).values

            if c_==100:
                c = c[-1]
            else:
                c = c_
        else:
            c = c[-1]
        
        if self.mode == "clr3":
            close_id = np.random.choice(self.clr_indices[c], 1).item()
            far_cl = (np.random.choice(len(self.clr_indices) - 1) + c + 1) % len(self.clr_indices)
            far_id = np.random.choice(self.clr_indices[far_cl], 1).item()

            print(close_id, far_id)
            close_ts, close_c = self.stsds[close_id]
            far_ts, far_c = self.stsds[far_id]

            # element 0 and 1 belong to the same class, element 2 belongs to another class
            return {"series": torch.stack([ts, close_ts, far_ts]), "label": c}

        if self.mode == "gasf":
            transformed = gaf_compute(ts, "s", (-1, 1))
            return {"series": ts, "label": c, "transformed": transformed}

        elif self.mode == "gadf":
            transformed = gaf_compute(ts, "d", (-1, 1))
            return {"series": ts, "label": c, "transformed": transformed}

        elif self.mode == "mtf":
            transformed = mtf_compute(ts, self.mtf_bins, (-1, 1))
            return {"series": ts, "label": c, "transformed": transformed}
        
        elif self.mode == "fft":
            transformed = torch.fft.fft(ts, dim=-1)
            transformed = torch.cat([transformed.real, transformed.imag], dim=0)
            return {"series": ts, "label": c, "transformed": transformed}

        elif self.mode == "cwt_test":
            transformed = pywt.cwt(ts.numpy(), scales=np.arange(1, ts.shape[1]//2, dtype=np.float64), sampling_period=1, wavelet="morl")[0]
            transformed = torch.from_numpy(transformed)
            transformed = transformed.permute(1, 0, 2)
            return {"series": ts, "label": c, "transformed": transformed}

        else:
            return {"series": ts, "label": c}
    
    def __del__(self):
        del self.stsds


class LSTSDataset(LightningDataModule):

    """ Data module for the experiments. """

    def __init__(self,
            stsds: STSDataset,    
            data_split: dict, batch_size: int, 
            random_seed: int = 42, 
            num_workers: int = 1,
            reduce_train_imbalance: bool = False,
            label_mode: int = 1,
            mode: str = None,
            mtf_bins: int = 50,
            skip: int = 1
            ) -> None:

        # save parameters as attributes
        super().__init__()
        
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers

        self.stsds = stsds
        self.wdw_len = self.stsds.wsize
        self.wdw_str = self.stsds.wstride
        self.sts_str = False

        self.l_patterns = None

        # gather dataset info   
        self.n_dims = self.stsds.STS.shape[0]
        self.n_classes = np.sum(np.unique(self.stsds.SCS)!=100).item()
        self.n_patterns = self.n_classes

        # convert to tensors
        if not torch.is_tensor(self.stsds.STS):
            self.stsds.STS = torch.from_numpy(self.stsds.STS).to(torch.float32)
        if not torch.is_tensor(self.stsds.SCS):
            self.stsds.SCS = torch.from_numpy(self.stsds.SCS).to(torch.int64)

        total_observations = self.stsds.indices.shape[0]
        train_indices = np.arange(total_observations)[data_split["train"](self.stsds.indices)]
        test_indices = np.arange(total_observations)[data_split["test"](self.stsds.indices)][::skip]
        val_indices = np.arange(total_observations)[data_split["val"](self.stsds.indices)][::skip]

        self.reduce_train_imbalance = reduce_train_imbalance

        if reduce_train_imbalance:
            train_labels = self.stsds.SCS[self.stsds.indices[train_indices]]
            train_label_weights = np.empty_like(train_labels, dtype=np.float32)

            cl, counts = torch.unique(train_labels, return_counts=True)
            for i in range(cl.shape[0]):
                train_label_weights[train_labels == cl[i]] = 1 / counts[i]

            examples_per_epoch = int(counts.float().mean().ceil().item())

            # add change points to the training indices (a change point up to 2/3 of the leading points in the time series)
            train_changePoints = self.stsds.getChangePointIndex()
            train_changePoints = np.tile(train_changePoints, (int(2*self.wdw_len/3), 1))
            for i in range(train_changePoints.shape[0]):
                train_changePoints[i, :] += i

            train_changePoints = train_changePoints[data_split["train"](train_changePoints)]

            train_indices = torch.cat([torch.from_numpy(train_indices), torch.from_numpy(train_changePoints)])
            train_label_weights = torch.cat(
                [torch.from_numpy(train_label_weights), torch.full_like(torch.from_numpy(train_changePoints), 1/train_changePoints.shape[0])])

            print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
            self.train_sampler = WeightedRandomSampler(train_label_weights, int(counts.float().mean().ceil().item()), replacement=True)

        if mode == "clr3":
            window_id, window_lb = self.stsds.getSameClassWindowIndex()
            window_id = window_id
            window_lb = window_lb.numpy()

            window_lb = window_lb[data_split["train"](window_id)]
            window_id = window_id[data_split["train"](window_id)]

            print(window_lb, np.unique(window_lb))

            clr_indices = []
            for cl in np.unique(window_lb):
                clr_indices.append(window_id[window_lb==cl])

        self.ds_train = StreamingTimeSeriesCopy(
            self.stsds, train_indices, label_mode, mode, mtf_bins, clr_indices)
        self.ds_test = StreamingTimeSeriesCopy(self.stsds, test_indices, label_mode, mode, mtf_bins)
        self.ds_val = StreamingTimeSeriesCopy(self.stsds, val_indices, label_mode, mode, mtf_bins)
        
    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        if self.reduce_train_imbalance:
            return DataLoader(self.ds_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, sampler=self.train_sampler,
                pin_memory=True, persistent_workers=True)
        else:
            return DataLoader(self.ds_train, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=True ,
                pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_val, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)
    
    def predict_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size, 
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)