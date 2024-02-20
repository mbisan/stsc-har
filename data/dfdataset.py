import os

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transforms.odtw import compute_oDTW, compute_oDTW_channel
import torch
from pytorch_lightning import LightningDataModule

# standard library imports
import multiprocessing as mp
import numpy as np

from data.base import STSDataset

import hashlib

class DFDataset(Dataset):
    def __init__(self, 
            stsds: STSDataset = None,
            patterns: np.ndarray = None,
            rho: float = 0.1,
            dm_transform = None,
            cached: bool = True,
            dataset_name: str = "") -> None:
        super().__init__()

        '''
            patterns: shape (n_shapes, channels, pattern_size)
        '''

        self.stsds = stsds
        self.cached = cached

        if not patterns.flags.c_contiguous:
            patterns = patterns.copy(order="c")

        self.patterns = patterns
        self.dm_transform = dm_transform

        self.n_patterns = self.patterns.shape[0] if len(self.patterns.shape) == 3 else self.patterns.shape[0] * self.stsds.STS.shape[0]
        if not self.stsds.feature_group is None and len(self.patterns.shape) == 3:
            self.n_patterns *= len(self.stsds.feature_group)

        self.rho = rho

        self.DM = []

        hash = hashlib.sha1(patterns.data)
        cache_id = f"{dataset_name}_" + hash.hexdigest()
        self.cache_dir = os.path.join(os.getcwd(), "cache_" + cache_id)
        print("hash of computed patterns:", cache_id)

        if cached:
            if not os.path.exists(self.cache_dir):
                os.mkdir(self.cache_dir)
            elif len(os.listdir(self.cache_dir)) == len(self.stsds.splits):
                print("Loading cached dissimilarity frames if available...")

            with open(os.path.join(self.cache_dir, "pattern.npz"), "wb") as f:
                np.save(f, self.patterns)

            for s in range(self.stsds.splits.shape[0] - 1):
                save_path = os.path.join(self.cache_dir, f"part{s}.npz")
                if not os.path.exists(save_path):
                    self._compute_dm(patterns, self.stsds.splits[s:s+2], save_path)
        
        else: # i.e. not cached
            for s in range(self.stsds.splits.shape[0] - 1):
                DM = self._compute_dm(patterns, self.stsds.splits[s:s+2], save_path=None)
                self.DM.append(DM)

        self.id_to_split = np.searchsorted(self.stsds.splits, self.stsds.indices) - 1

    def _compute_dm(self, pattern, split, save_path):
        if (not self.stsds.feature_group is None) and len(pattern.shape) == 3:
            DM = self._compute_dm_groups(pattern, split, save_path)

        else:

            if len(pattern.shape) == 3:
                DM = compute_oDTW(self.stsds.STS[:, split[0]:split[1]], pattern, rho=self.rho)
            elif len(pattern.shape) == 2:
                DM = compute_oDTW_channel(self.stsds.STS[:, split[0]:split[1]], pattern, rho=self.rho)

        # put time dimension in the first dimension
        DM = np.ascontiguousarray(np.transpose(DM, (2, 0, 1)))
        # therefore, DM has dimensions (n, num_frames, patt_len)

        if save_path is None:
            return DM
        else:
            with open(save_path, "wb") as f:
                np.save(f, DM)

    def _compute_dm_groups(self, pattern, split, save_path):
        DM_groups = []
        for group in self.stsds.feature_group:
            DM = compute_oDTW(self.stsds.STS[group, split[0]:split[1]], pattern[:,group,:], rho=self.rho)

            DM_groups.append(DM)
        
        return np.concatenate(DM_groups, axis=0)

    def __len__(self):
        return len(self.stsds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:

        id = self.stsds.indices[index]

        # identify the split of the index
        s = self.id_to_split[index]
        first = id - self.stsds.wsize*self.stsds.wstride - self.stsds.splits[s] + 1
        last = id - self.stsds.splits[s] + 1

        if self.cached:
            dm_np = np.load(os.path.join(self.cache_dir, f"part{s}.npz"), mmap_mode="r")[first:last:self.stsds.wstride].copy()
        else:
            dm_np = self.DM[s][first:last:self.stsds.wstride].copy()
        dm = torch.permute(torch.from_numpy(dm_np), (1, 2, 0)) # recover the dimensions of dm (n_frames, patt_len, n)

        if not self.dm_transform is None:
            dm = self.dm_transform(dm)

        ts, c = self.stsds[index]

        return (dm, ts, c)

class DFDatasetCopy(Dataset):
    def __init__(self,
            dfds: DFDataset, indices: np.ndarray, label_mode: int = 1) -> None:
        super().__init__()

        assert label_mode%2==1

        self.dfds = dfds
        self.indices = indices
        self.label_mode = label_mode
        
    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, int]:

        df, ts, c = self.dfds[self.indices[index]]

        if self.label_mode > 1:
            c = torch.mode(c[-self.label_mode:]).values
        else:
            c = c[-1]

        return {"frame": df, "series": ts, "label": c}
    
    def __del__(self):
        del self.dfds

class LDFDataset(LightningDataModule):

    """ Data module for the experiments. """

    def __init__(self,
            dfds: DFDataset,    
            data_split: dict, 
            batch_size: int, 
            random_seed: int = 42, 
            num_workers: int = mp.cpu_count()//2,
            reduce_train_imbalance: bool = False,
            label_mode: int = 1,
            ) -> None:

        '''
            dfds: Dissimilarity frame DataSet
            data_split: How to split the dfds, example below 

            data_split = {
                "train" = lambda indices: train_condition,
                "val" = lambda indices: val_condition,
                "test" = lambda indices: test_condition
            }
        '''

        # save parameters as attributes
        super().__init__()
        
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers

        self.dfds = dfds
        self.wdw_len = self.dfds.stsds.wsize
        self.wdw_str = self.dfds.stsds.wstride
        self.sts_str = False

        # gather dataset info   
        self.n_dims = self.dfds.stsds.STS.shape[0]
        self.n_classes = len(np.unique(self.dfds.stsds.SCS))
        self.n_patterns = self.dfds.n_patterns
        self.l_patterns = self.dfds.patterns.shape[-1]

        # convert to tensors
        if not torch.is_tensor(self.dfds.stsds.STS):
            self.dfds.stsds.STS = torch.from_numpy(self.dfds.stsds.STS).to(torch.float32)
        if not torch.is_tensor(self.dfds.stsds.SCS):
            self.dfds.stsds.SCS = torch.from_numpy(self.dfds.stsds.SCS).to(torch.int64)

        total_observations = self.dfds.stsds.indices.shape[0]
        train_indices = np.arange(total_observations)[data_split["train"](self.dfds.stsds.indices)]
        test_indices = np.arange(total_observations)[data_split["test"](self.dfds.stsds.indices)]
        val_indices = np.arange(total_observations)[data_split["val"](self.dfds.stsds.indices)]

        self.reduce_train_imbalance = reduce_train_imbalance
        if reduce_train_imbalance:
            self.train_labels = self.dfds.stsds.SCS[self.dfds.stsds.indices[train_indices]]
            self.train_label_weights = np.empty_like(self.train_labels, dtype=np.float32)

            cl, counts = torch.unique(self.train_labels, return_counts=True)
            for i in range(cl.shape[0]):
                self.train_label_weights[self.train_labels == cl[i]] = self.train_labels.shape[0] / counts[i]

            examples_per_epoch = int(counts.float().mean().ceil().item())
            print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
            self.train_sampler = WeightedRandomSampler(self.train_label_weights, int(counts.float().mean().ceil().item()), replacement=True)

        self.ds_train = DFDatasetCopy(self.dfds, train_indices, label_mode)
        self.ds_test = DFDatasetCopy(self.dfds, test_indices, label_mode)
        self.ds_val = DFDatasetCopy(self.dfds, val_indices, label_mode)
        
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