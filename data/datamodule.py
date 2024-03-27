from typing import List

import numpy as np

from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from pytorch_lightning import LightningDataModule, seed_everything

from data.stsdataset import STSDataset
from data.dfdataset import DFDataset, PatternConf
from data.har.har_datasets import load_dataset

from data.har.label_mappings import mappings


class CustomRandomSampler(Sampler):

    '''
        From a list of lists of indices (one list of indices per class)
        selects all the indices from the minority class, and randomly selects
        this same number of indices in each of the remaining classes.
    '''

    def __init__(self, indices_per_class):
        super().__init__()

        self.indices_per_class = indices_per_class
        self.obs_per_class = min(len(indices) for indices in self.indices_per_class)

    def __iter__(self):
        selected = []
        for indices in self.indices_per_class:
            selected.append(np.random.choice(indices, self.obs_per_class, replace=False))
        selected = np.concatenate(selected)
        np.random.shuffle(selected)
        return iter(selected.tolist())

    def __len__(self):
        return self.obs_per_class * len(self.indices_per_class)

class STSDataModule(LightningDataModule):

    """ Data module for the experiments. """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
            dataset_name: str,
            dataset_dir: str,
            window_size: int,
            window_stride: int,
            batch_size: int,
            random_seed: int,
            num_workers: int,
            label_mode: int = 1,
            reduce_imbalance: bool = True,
            same_class: bool = False,
            subjects_for_test: List[int] = None,
            n_val_subjects: int = 2,
            overlap: int = -1,
            patterns = PatternConf(None, None, .1, False, 300),
            pattern_stride: int = 1,
            triplets: bool = False
            ) -> None:
        # pylint: disable=too-many-arguments too-many-locals

        # save parameters as attributes
        super().__init__()

        seed_everything(random_seed)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wdw_len = window_size
        self.wdw_str = window_stride

        dataset = load_dataset(dataset_dir)

        val_subjects = np.random.choice(
            [i for i in range(len(dataset)) if not i in subjects_for_test], n_val_subjects).tolist()
        not_train = val_subjects + subjects_for_test

        if not patterns.pattern_type is None:
            self.train_dataset = DFDataset(
                patterns=patterns,
                data=[data for i, data in enumerate(dataset) if not i in not_train],
                wsize=self.wdw_len,
                wstride=self.wdw_str,
                label_mapping=mappings[dataset_name],
                label_mode=label_mode,
                triplets=triplets,
                pattern_stride=pattern_stride
            )
            self.val_dataset = DFDataset(
                data=[data for i, data in enumerate(dataset) if i in val_subjects],
                wsize=self.wdw_len,
                wstride=self.wdw_str,
                minmax=self.train_dataset.minmax,
                label_mapping=mappings[dataset_name],
                label_mode=label_mode,
                computed_patterns=self.train_dataset.patterns,
                pattern_stride=pattern_stride
            )
            self.test_dataset = DFDataset(
                data=[data for i, data in enumerate(dataset) if i in subjects_for_test],
                wsize=self.wdw_len,
                wstride=self.wdw_str,
                minmax=self.train_dataset.minmax,
                label_mapping=mappings[dataset_name],
                label_mode=label_mode,
                computed_patterns=self.train_dataset.patterns,
                pattern_stride=pattern_stride
            )
            self.l_patterns = self.train_dataset.DM[0].shape[-1]
        else:
            self.train_dataset = STSDataset(
                data=[data for i, data in enumerate(dataset) if not i in not_train],
                wsize=self.wdw_len,
                wstride=self.wdw_str,
                label_mapping=mappings[dataset_name],
                label_mode=label_mode,
                triplets=triplets,
            )
            self.val_dataset = STSDataset(
                data=[data for i, data in enumerate(dataset) if i in val_subjects],
                wsize=self.wdw_len,
                wstride=self.wdw_str,
                minmax=self.train_dataset.minmax,
                label_mapping=mappings[dataset_name],
                label_mode=label_mode,
            )
            self.test_dataset = STSDataset(
                data=[data for i, data in enumerate(dataset) if i in subjects_for_test],
                wsize=self.wdw_len,
                wstride=self.wdw_str,
                minmax=self.train_dataset.minmax,
                label_mapping=mappings[dataset_name],
                label_mode=label_mode,
            )
            self.l_patterns = self.train_dataset.stream.shape[1] # i.e. self.n_dims

        # gather dataset info
        self.n_dims = self.train_dataset.stream.shape[1]
        self.n_classes = np.sum(np.unique(self.train_dataset.labels)!=100).item()
        if hasattr(self.train_dataset, "n_patterns"):
            self.n_patterns = self.train_dataset.n_patterns
        else:
            self.n_patterns = self.n_dims

        self.reduce_train_imbalance = reduce_imbalance

        if reduce_imbalance:
            weights = self.train_dataset.reduce_imbalance()
        elif same_class:
            weights = self.train_dataset.set_same_class_indices()
        else:
            weights = None

        self.train_dataset.indices_per_class()

        if overlap>=0:
            self.test_dataset.apply_overlap(overlap)
            self.val_dataset.apply_overlap(overlap)

        if self.reduce_train_imbalance:
            self.train_sampler = WeightedRandomSampler(
                weights,
                num_samples=min(
                        min(len(c) for c in self.train_dataset.per_class), 15000
                    )*self.n_classes,
                replacement=False)
            print(f"Sampling {self.train_sampler.num_samples} balanced samples")

    def train_dataloader(self) -> DataLoader:
        """ Returns the training DataLoader. """
        if self.reduce_train_imbalance:
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, sampler=self.train_sampler,
                pin_memory=True, persistent_workers=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True ,
            pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """ Returns the validation DataLoader. """
        return DataLoader(self.val_dataset, batch_size=self.batch_size*4,
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """ Returns the test DataLoader. """
        return DataLoader(self.test_dataset, batch_size=self.batch_size*4,
            num_workers=self.num_workers, shuffle=False,
            pin_memory=True, persistent_workers=True)
