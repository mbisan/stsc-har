import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

from data.base import STSDataset

class StreamingTimeSeriesCopy(Dataset):

    def __init__(self,
            stsds: STSDataset, indices: np.ndarray, delay: int = 0
            ) -> None:
        super().__init__()

        self.stsds = stsds
        self.indices = indices
        self.delay = delay
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.delay:
            id = min(self.indices[index] + np.random.randint(low=-self.delay, high=self.delay), len(self.stsds))
            ts, c = self.stsds[id]
        else:
            ts, c = self.stsds[self.indices[index]]
        return {"series": ts, "scs": c}
    
    def __del__(self):
        del self.stsds


class LSegDataset(LightningDataModule):

    """ Data module for the experiments. """

    def __init__(self,
            stsds: STSDataset,    
            data_split: dict, batch_size: int, 
            random_seed: int = 42, 
            num_workers: int = 1,
            reduce_train_imbalance: bool = False,
            skip = 1
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
        self.n_classes = len(np.unique(self.stsds.SCS))
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

        self.train_labels = self.stsds.SCS[self.stsds.indices[train_indices]]

        label_diff = np.diff(self.train_labels)
        label_diff[np.random.randn(label_diff.shape[0]) > 2.5] = 1
        train_points = np.nonzero(label_diff)[0]

        self.ds_train = StreamingTimeSeriesCopy(self.stsds, train_points + int(self.wdw_len/2), delay=self.wdw_len)
        self.ds_test = StreamingTimeSeriesCopy(self.stsds, test_indices)
        self.ds_val = StreamingTimeSeriesCopy(self.stsds, val_indices)
        
    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
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