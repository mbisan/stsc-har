
import numpy as np
import torch

from data.base import STSDataset

from torch.utils.data import WeightedRandomSampler

def reduce_imbalance(train_indices, stsds: STSDataset, train_split, include_change_points = True):

    train_labels = stsds.SCS[stsds.indices[train_indices]]
    train_label_weights = np.empty_like(train_labels, dtype=np.float32)

    cl, counts = torch.unique(train_labels, return_counts=True)
    for i in range(cl.shape[0]):
        train_label_weights[train_labels == cl[i]] = 1 / counts[i]

    examples_per_epoch = int(counts.float().mean().ceil().item())

    # add change points to the training indices (a change point up to 2/3 of the leading points in the time series)
    if include_change_points:
        train_changePoints = stsds.getChangePointIndex()
        train_changePoints = np.tile(train_changePoints, (int(2*stsds.wsize/3), 1)) + int(stsds.wsize/3)
        for i in range(train_changePoints.shape[0]):
            train_changePoints[i, :] += i

        train_changePoints = train_changePoints[train_split(stsds.indices[train_changePoints])]

        train_indices = torch.cat([torch.from_numpy(train_indices), torch.from_numpy(train_changePoints)])
        train_label_weights = torch.cat(
            [torch.from_numpy(train_label_weights), torch.full_like(torch.from_numpy(train_changePoints), 1/train_changePoints.shape[0])])

    print(f"Sampling {examples_per_epoch} (balanced) observations per epoch.")
    train_sampler = WeightedRandomSampler(train_label_weights, examples_per_epoch, replacement=True)

    return train_indices, train_sampler