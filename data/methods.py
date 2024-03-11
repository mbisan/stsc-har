
import numpy as np
from torch.utils.data import WeightedRandomSampler

from data.base import STSDataset

def reduce_imbalance(train_indices, stsds: STSDataset, _, include_change_points = True):

    train_labels = stsds.SCS[stsds.indices[train_indices]]
    train_label_weights = np.empty_like(train_labels, dtype=np.float32)

    cl, counts = np.unique(train_labels, return_counts=True)
    for i in range(cl.shape[0]):
        train_label_weights[train_labels == cl[i]] = 1 / counts[i]

    examples_per_epoch = int(np.min(counts))*cl.shape[0]

    # add change points to the training indices
    # (a change point from 1/3 up to 2/3 of the leading points in the time series)
    if include_change_points:
        train_change_points = stsds.getChangePointIndex()
        train_change_points = np.tile(
            train_change_points, (int(stsds.wsize/3), 1)) + int(stsds.wsize/3)
        for i in range(train_change_points.shape[0]):
            train_change_points[i, :] += i

        train_change_points = np.intersect1d(train_indices, train_change_points)

        if train_change_points.shape[0] > 0:
            train_indices = np.concatenate([train_indices, train_change_points])
            train_label_weights = np.concatenate(
                [train_label_weights,
                 np.full_like(train_change_points, 1/train_change_points.shape[0])])

    print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
    train_sampler = WeightedRandomSampler(
        train_label_weights, examples_per_epoch, replacement=False)

    return train_indices, train_sampler
