import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

from data.base import STSDataset

class StreamingTimeSeriesCopy(Dataset):

    def __init__(self,
            stsds: STSDataset, indices: np.ndarray
            ) -> None:
        super().__init__()

        self.stsds = stsds
        self.indices = indices
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:
        ts, c = self.stsds[self.indices[index]]
        return {"series": ts, "label": torch.mode(c).values}
    
    def __del__(self):
        del self.stsds


class LConDataset(LightningDataModule):

    """ Data module for the experiments. """

    def __init__(self,
            stsds: STSDataset,    
            data_split: dict, batch_size: int, 
            random_seed: int = 42, 
            num_workers: int = 1
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

        sameClassIndex, sameClassClass = self.stsds.getSameClassWindowIndex()
        train_indices = sameClassIndex[data_split["train"](sameClassIndex)]
        test_indices = sameClassIndex[data_split["test"](sameClassIndex)]
        val_indices = sameClassIndex[data_split["val"](sameClassIndex)]

        sameClassClass = sameClassClass[data_split["train"](sameClassIndex)]
        self.train_label_weights = np.empty_like(sameClassClass, dtype=np.float32)

        cl, counts = torch.unique(sameClassClass, return_counts=True)
        for i in range(cl.shape[0]):
            self.train_label_weights[sameClassClass == cl[i]] = sameClassClass.shape[0] / counts[i]

        examples_per_epoch = int(counts.float().mean().ceil().item())
        print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
        self.train_sampler = WeightedRandomSampler(self.train_label_weights, int(counts.float().mean().ceil().item()), replacement=True)

        self.ds_train = StreamingTimeSeriesCopy(self.stsds, train_indices)
        self.ds_test = StreamingTimeSeriesCopy(self.stsds, test_indices)
        self.ds_val = StreamingTimeSeriesCopy(self.stsds, val_indices)
        
    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        return DataLoader(self.ds_train, batch_size=self.batch_size, 
            num_workers=self.num_workers, sampler=self.train_sampler,
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