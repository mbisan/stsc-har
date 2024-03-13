import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import pywt

from data.base import STSDataset
from data.methods import reduce_imbalance, reduce_imbalance2

from transforms.gaf_mtf import mtf_compute, gaf_compute

class StreamingTimeSeriesCopy(Dataset):

    def __init__(self,
            stsds: STSDataset,
            indices: np.ndarray,
            label_mode: int = 1,
            mode: str = None,
            mtf_bins: int = 30,
            clr_indices: list[np.ndarray] = None
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
                c_ = c[-1]
        else:
            c_ = c[-1]

        if (self.mode in ["clr3", "clr"]) and not self.clr_indices is None:
            close_id = np.random.choice(self.clr_indices[c_], 1).item()
            far_cl = (np.random.choice(len(self.clr_indices) - 1) + c_ + 1) % len(self.clr_indices)

            # el sampleo del negativo (far_cl) está bien, cualquier carencia es debido al modelo
            far_id = np.random.choice(self.clr_indices[far_cl], 1).item()

            close_ts, _ = self.stsds[close_id]
            far_ts, _ = self.stsds[far_id]

            # element 0 and 1 belong to the same class, element 2 belongs to another class
            return {"series": ts, "label": c_,
                    "triplet": torch.stack([ts, close_ts, far_ts]),
                    "far_cl": far_cl,
                    "change_point": c.unique().shape[0] > 1}

        elif self.mode in ["clr_ssl"]:
            print(self.indices[(index-self.stsds.wsize):(index+self.stsds.wsize)].shape)
            close_id = np.random.choice(
                self.indices[(index-self.stsds.wsize):(index+self.stsds.wsize)], 1).item()

            far_id = np.random.choice(
                np.concatenate([
                    self.indices[(index-2*self.stsds.wsize):],
                    self.indices[(index+2*self.stsds.wsize):]
                ]), 1).item()

            close_ts, _ = self.stsds[close_id]
            far_ts, _ = self.stsds[far_id]

            # element 0 and 1 belong to the same class, element 2 belongs to another class
            return {"series": ts, "label": c_,
                    "triplet": torch.stack(
                        # adding some noise so that ts and close_ts are never the same
                        [ts, close_ts + 0.01 * np.random.randn(close_ts.shape), far_ts]),
                    "far_cl": far_cl,
                    "change_point": c.unique().shape[0] > 1}

        elif self.mode == "gasf":
            transformed = gaf_compute(ts, "s", (-1, 1))
            return {"series": ts, "label": c_, "transformed": transformed}

        elif self.mode == "gadf":
            transformed = gaf_compute(ts, "d", (-1, 1))
            return {"series": ts, "label": c_, "transformed": transformed}

        elif self.mode == "mtf":
            transformed = mtf_compute(ts, self.mtf_bins, (-1, 1))
            return {"series": ts, "label": c_, "transformed": transformed}

        elif self.mode == "fft":
            # pylint: disable=not-callable
            transformed = torch.fft.fft(ts, dim=-1)
            transformed = torch.cat([transformed.real, transformed.imag], dim=0)
            return {"series": ts, "label": c_, "transformed": transformed}

        elif self.mode == "cwt_test":
            transformed = pywt.cwt(
                ts.numpy(), scales=np.arange(1, ts.shape[1]//2, dtype=np.float64),
                sampling_period=1, wavelet="morl")[0]
            transformed = torch.from_numpy(transformed)
            transformed = transformed.permute(1, 0, 2)
            return {"series": ts, "label": c_, "transformed": transformed}

        elif self.mode == "ts":
            return {"series": ts, "label": c_}

        # change_point is true if the window contains more than one class
        return {"series": ts, "label": c_, "scs": c, "change_point": c.unique().shape[0] > 1}

    def __del__(self):
        del self.stsds


class LSTSDataset(LightningDataModule):

    """ Data module for the experiments. """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
            stsds: STSDataset,
            data_split: dict, batch_size: int,
            random_seed: int = 42,
            num_workers: int = 1,
            reduce_train_imbalance: bool = False,
            label_mode: int = 1,
            mode: str = None,
            mtf_bins: int = 50,
            skip: int = 1,
            same_class: bool = False,
            change_points: bool = True
            ) -> None:
        # pylint: disable=too-many-arguments too-many-locals

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

        total_observations = self.stsds.indices.shape[0]
        train_indices = np.arange(total_observations)[data_split["train"](self.stsds.indices)]

        if same_class:
            train_indices = np.intersect1d(train_indices, self.stsds.getSameClassWindowIndex()[0])

        test_indices = np.arange(total_observations)[data_split["test"](self.stsds.indices)][::skip]
        val_indices = np.arange(total_observations)[data_split["val"](self.stsds.indices)][::skip]

        self.reduce_train_imbalance = reduce_train_imbalance

        if reduce_train_imbalance:
            train_indices, train_sampler = reduce_imbalance2(
                train_indices, self.stsds, data_split["train"], include_change_points=change_points)
            self.train_sampler = train_sampler

        clr_indices = None
        if mode in ["clr3", "clr"]:
            clr_indices = self.stsds.getIndicesByClass(data_split["train"])

        self.stsds.toTensor()

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
        return DataLoader(self.ds_train, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True ,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """ Returns the validation DataLoader. """
        return DataLoader(self.ds_val, batch_size=self.batch_size*4,
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size*4,
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.ds_test, batch_size=self.batch_size*4,
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)
