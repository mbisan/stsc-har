from typing import List, Tuple, Dict

import numpy as np
from scipy import stats

from torch.utils.data import Dataset

# pylint: disable=invalid-name too-many-instance-attributes

class STSDataset(Dataset):

    wsize: int
    wstride: int
    label_mode: int
    get_triplets: bool

    splits: np.ndarray
    stream: np.ndarray # of shape (n, n_dims)
    labels: np.ndarray
    indices: np.ndarray

    minmax: Tuple[np.ndarray, np.ndarray]

    feature_group: List[np.ndarray]
    label_mapping: np.ndarray

    id_to_split: np.ndarray

    per_class: List[np.ndarray]

    def __init__(self,
            data: List[List[Tuple[np.ndarray, np.ndarray]]],
            wsize: int = 10,
            wstride: int = 1,
            minmax: Tuple[np.ndarray, np.ndarray] = None,
            label_mapping: np.ndarray = np.arange(256, dtype=np.int64),
            label_mode: int = 1,
            feature_group: List[np.ndarray] = None,
            triplets: bool = False
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

            indices are initialized to all the possible windows on every sts
        '''
        super().__init__()

        self.wsize = wsize
        self.wstride = wstride
        self.label_mode = label_mode
        self.get_triplets = triplets

        # set splits, stream and labels. Segments shorter than wsize are removed
        splits = np.concatenate(
            [np.array([sts.shape[0] for sts, _ in user if sts.shape[0] > wsize]) for user in data])
        self.splits = np.zeros(splits.shape[0] + 1, dtype=np.int64)
        self.splits[1:] = np.cumsum(splits)

        self.stream = np.concatenate(
            [np.concatenate([sts for sts, _ in user if sts.shape[0] > wsize]) for user in data]
        )

        self.labels = label_mapping[np.concatenate(
            [np.concatenate(
                [lbl.astype(np.int64) for _, lbl in user if lbl.shape[0] > wsize]) for user in data]
        ).reshape(-1)]

        self.indices = np.array([])

        if minmax is None:
            minmax = (
                np.mean(self.stream, axis=0, keepdims=True),
                np.std(self.stream, axis=0, keepdims=True)
            )
        self.minmax = minmax

        # minmax scale the stream
        self.stream -= self.minmax[0]
        self.stream /= self.minmax[1] + 1e-6

        # dict with groups of features i.e. [np.array([0, 1, 2]), np.array([3, 4, 5])]
        self.feature_group = feature_group

        self.label_mapping = label_mapping

        self.indices = None
        self.id_to_split = np.searchsorted(self.splits, np.arange(self.labels.shape[0])) - 1
        self.set_indices() # init indices

        self.per_class: List[np.ndarray] = None

    def set_indices(self) -> None:
        # initialized self.indices
        self.indices = np.arange(self.labels.shape[0], dtype=np.int64)

        # remove initial and unlabeled window indices
        for i in range(self.wsize * self.wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices[self.labels == 100] = 0 # remove observations with no label
        self.indices = self.indices[np.nonzero(self.indices)]

    def set_same_class_indices(self) -> np.ndarray:
        # sets self.indices with windows of the same class
        same_class = self.same_class_window()
        window_labels = self.labels[same_class]
        self.indices = same_class

        weights = np.empty_like(same_class, dtype=np.float64)

        unique, counts = np.unique(window_labels, return_counts=True)
        assert not 100 in unique

        for i, c in enumerate(unique):
            weights[window_labels == c] = 1 / counts[i]

        return weights

    def points_to_remove(self) -> np.ndarray:
        splits_remove = np.tile(self.splits, (self.wsize, 1))
        for i in range(self.wsize):
            splits_remove[i] += i
        return splits_remove

    def reduce_imbalance(self) -> np.ndarray:
        # sets self.indices and returns the weights for each class
        same_class_windows = self.same_class_window()
        window_labels = self.labels[same_class_windows]
        change_points = self.change_points()

        change_points = np.tile(change_points, (3, 1))
        for i in range(3):
            change_points[i] += (i+1)*(self.wsize//6)
        change_points = change_points.reshape(-1)

        change_points = change_points[self.labels[change_points] != 100]
        change_points = change_points[~np.isin(change_points, self.points_to_remove())]

        weights = np.empty_like(same_class_windows, dtype=np.float64)

        unique, counts = np.unique(window_labels, return_counts=True)
        assert not 100 in unique

        for i, c in enumerate(unique):
            weights[window_labels == c] = 1 / counts[i]

        indices = np.concatenate([same_class_windows, change_points])
        weights = np.concatenate(
            [weights, np.full_like(change_points, 1/change_points.shape[0], dtype=np.float64)])

        self.indices = indices

        return weights

    def same_class_window(self) -> np.ndarray:
        # returns array with positions in the sts with same class windows
        diff = np.diff(self.labels)
        ids = np.concatenate(([0], np.nonzero(diff)[0], [self.stream.shape[0]]))

        temp_indices = np.zeros_like(self.labels, dtype=np.bool_)

        offset = self.wsize*self.wstride + 1
        for i in range(ids.shape[0]-1):
            if ids[i+1] - ids[i] >= offset:
                temp_indices[(ids[i] + offset):(ids[i+1]+1)] = True

        indices = np.nonzero(temp_indices)[0]
        valid = indices[self.labels[indices] != 100]

        valid = valid[~np.isin(valid, self.points_to_remove())]

        return valid

    def change_points(self) -> np.ndarray:
        # returns array with positions in the sts with change points
        diff = np.diff(self.labels)
        indices = np.nonzero(diff)[0]
        indices = indices[self.labels[indices] != 100]

        valid = indices[~np.isin(indices, self.points_to_remove())]

        return valid

    def indices_per_class(self) -> None:
        # sets the per_class variable to the indices of each class
        unique = np.unique(self.labels[self.indices])
        assert not 100 in unique

        self.per_class = []
        for c in unique:
            indices = (self.labels == c).nonzero()[0]
            self.per_class.append(indices)

    def apply_overlap(self, overlap: int) -> None:
        self.indices = self.indices[::(self.wsize-overlap)]

    def __len__(self) -> int:
        return self.indices.shape[0]

    def triplet(self, c: int) -> Tuple[np.ndarray, np.ndarray]:
        # get the close and far samples for triplet/contrastive learning

        close_id = np.random.choice(self.per_class[c], 1).item()
        far_cl = (np.random.choice(len(self.per_class) - 1) + c + 1) % len(self.per_class)

        # el sampleo del negativo (far_cl) está bien, cualquier carencia es debido al modelo
        far_id = np.random.choice(self.per_class[far_cl], 1).item()

        close_first = close_id - self.wsize*self.wstride + self.wstride
        close_last = close_id + 1

        far_first = far_id - self.wsize*self.wstride + self.wstride
        far_last = far_id + 1

        return (
            self.stream[close_first:close_last:self.wstride].T,
            self.stream[far_first:far_last:self.wstride].T
        )

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        pos = self.indices[index]

        first = pos - self.wsize*self.wstride + self.wstride
        last = pos + 1

        scs = self.labels[first:last:self.wstride]
        if self.label_mode > 1:
            c = stats.mode(scs[-self.label_mode:])
        else:
            c = scs[-1]

        return {
            "series": self.stream[first:last:self.wstride].T,
            "scs": scs,
            "label": c,
            "triplet": self.triplet(c) if self.get_triplets else 0
        }
