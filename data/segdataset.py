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
            id = min(self.indices[index] + np.random.randint(low=self.delay, high=2*self.delay), len(self.stsds)-1)
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

        self.reduce_train_imbalance = reduce_train_imbalance
        if reduce_train_imbalance:

            change_points = self.stsds.getChangePointIndex()
            sameClassIndex, sameClassClass = self.stsds.getSameClassWindowIndex() # array with indices of the same class, and class of the index

            # get sameclass window indices among train indices
            sameclass_indices = sameClassIndex[data_split["train"](sameClassIndex)]
            sameclass_class = sameClassClass[data_split["train"](sameClassIndex)]
            sameclass_indices_weights = torch.empty_like(torch.from_numpy(sameclass_indices), dtype=torch.float32)

            cl, counts = torch.unique(sameclass_class, return_counts=True)
            for i in range(cl.shape[0]):
                sameclass_indices_weights[sameclass_class == cl[i]] = 1 / counts[i]

            examples_per_epoch = int(counts.float().mean().ceil().item()) + change_points.shape[0]

            train_indices = torch.cat([torch.from_numpy(sameclass_indices), torch.from_numpy(change_points)])
            train_indices_weights = torch.cat([sameclass_indices_weights, torch.full_like(torch.from_numpy(change_points), 1/change_points.shape[0])])

            print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
            self.train_sampler = WeightedRandomSampler(train_indices_weights, int(counts.float().mean().ceil().item()), replacement=True)
            # train_indices = reduce_imbalance(train_indices, self.train_labels, seed=random_seed)

        self.ds_train = StreamingTimeSeriesCopy(self.stsds, train_indices, delay=int(self.wdw_len/3))
        self.ds_test = StreamingTimeSeriesCopy(self.stsds, test_indices)
        self.ds_val = StreamingTimeSeriesCopy(self.stsds, val_indices)
        
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