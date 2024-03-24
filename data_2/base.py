from typing import List, Tuple

import numpy as np
import torch

from torch.utils.data import Dataset

# pylint: disable=invalid-name too-many-instance-attributes

class STSDataset(Dataset):

    def __init__(self,
            data: List[List[Tuple[np.ndarray, np.ndarray]]],
            wsize: int = 10,
            wstride: int = 1,
            minmax: Tuple[np.ndarray, np.ndarray] = None,
            feature_group: List[np.ndarray] = None,
            label_mapping: np.ndarray = np.arange(256, dtype=np.int64)
            ) -> None:
        '''
            Base class for STS dataset

            Inputs:
                data: list of lists, containing the pairs (sts, labels) for each subject
                      subjects are lost, by considering all sts together
                wsize: window size
                wstride: window stride
                minmax: minmax values to use to scale data
                feature_group: groups of features, i.e. for different sensors
        '''
        super().__init__()

        splits = np.concatenate(
            [np.array([sts.shape[0] for sts, _ in user]) for user in data])
        self.splits = np.zeros(splits.shape[0] + 1, dtype=np.int64)
        self.splits[1:] = np.cumsum(splits)

        self.stream = np.concatenate(
            [np.concatenate([sts for sts, _ in user]) for user in data]
        )

        self.labels = label_mapping[np.concatenate(
            [np.concatenate([lbl.astype(np.int64) for _, lbl in user]) for user in data]
        )]

        self.wsize = wsize
        self.wstride = wstride

        self.indices = np.array([])

        if minmax is None:
            minmax = (
                np.min(self.stream, axis=0, keepdims=True),
                np.max(self.stream, axis=0, keepdims=True)
            )
        self.minmax = minmax

        # minmax scale the stream
        self.stream -= self.minmax[0]
        self.stream /= (self.minmax[1] - self.minmax[0])

        # dict with groups of features i.e. [np.array([0, 1, 2]), np.array([3, 4, 5])]
        self.feature_group = feature_group

        self.indices = np.arange(self.labels.shape[0], dtype=np.int64)

        # remove initial and unlabeled window indices
        for i in range(self.wsize * self.wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices[self.labels == 100] = 0 # remove observations with no label
        self.indices = self.indices[np.nonzero(self.indices)]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        first = self.indices[index] - self.wsize*self.wstride + self.wstride
        last = self.indices[index] + 1

        return self.stream[first:last:self.wstride], self.labels[first:last:self.wstride]

    def getSameClassWindowIndex(self, return_mask=False):
        # returns array with positions in the indices with same class windows
        diff = np.diff(self.labels)
        ids = np.concatenate(([0], np.nonzero(diff)[0], [self.stream.shape[0]]))

        temp_indices = np.zeros_like(self.labels, dtype=np.bool_)

        offset = self.wsize*self.wstride
        for i in range(ids.shape[0]-1):
            if ids[i+1] - ids[i] >= offset:
                temp_indices[(ids[i] + offset):(ids[i+1]+1)] = True

        indices_new = temp_indices[self.indices]

        if return_mask:
            return indices_new

        sameClassWindowIndex = np.arange(self.indices.shape[0])[indices_new]

        return sameClassWindowIndex, self.labels[self.indices[sameClassWindowIndex]]

    def getChangePointIndex(self):
        # returns array with positions in the indices with change points
        diff = np.diff(self.labels)
        ids = np.nonzero(diff)[0]

        temp_indices = np.zeros_like(self.labels, dtype=np.bool_)
        temp_indices[ids] = True

        indices_new = temp_indices[self.indices]
        changePointIndex = np.arange(self.indices.shape[0])[indices_new]

        return changePointIndex

    def getIndicesByClass(self):
        window_id, window_lb = self.getSameClassWindowIndex()

        clr_indices = []
        for cl in np.unique(window_lb):
            clr_indices.append(window_id[window_lb==cl])

        return clr_indices
