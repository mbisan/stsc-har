import numpy as np

from torch.utils.data import Dataset

class STSDataset(Dataset):

    def __init__(self,
            wsize: int = 10,
            wstride: int = 1,
            ) -> None:
        super().__init__()

        '''
            Base class for STS dataset

            Inputs:
                wsize: window size
                wstride: window stride
        '''

        self.wsize = wsize
        self.wstride = wstride

        self.splits = None

        self.STS = None
        self.SCS = None

        self.indices = None

        self.feature_group = None # dict with groups of features i.e. [np.array([0, 1, 2]), np.array([3, 4, 5])]

    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        first = self.indices[index]-self.wsize*self.wstride+self.wstride
        last = self.indices[index]+1

        return self.STS[:, first:last:self.wstride], self.SCS[first:last:self.wstride]
    
    def position(self, index: int) -> tuple[np.ndarray, np.ndarray]:

        first = index-self.wsize*self.wstride+self.wstride
        last = index+1

        return self.STS[:, first:last:self.wstride], self.SCS[first:last:self.wstride]

    def sliceFromArrayOfIndices(self, indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(indexes.shape) == 1 # only accept 1-dimensional arrays

        return_sts = np.empty((indexes.shape[0], self.STS.shape[0], self.wsize))
        return_scs = np.empty((indexes.shape[0], self.wsize))

        for i, id in enumerate(indexes):
            ts, c = self[id]
            return_scs[i] = c
            return_sts[i] = ts

        return return_sts, return_scs
    
    def getSameClassWindowIndex(self, return_mask=False):
        diff = np.diff(self.SCS)
        ids = np.concatenate(([0], np.nonzero(diff)[0], [self.SCS.shape[0]]))

        temp_indices = np.zeros_like(self.SCS, dtype=np.bool_)

        offset = self.wsize*self.wstride + self.wstride
        for i in range(ids.shape[0]-1):
            if ids[i+1] - ids[i] >= offset:
                temp_indices[(ids[i] + offset):(ids[i+1]+1)] = True

        indices_new = temp_indices[self.indices]

        if return_mask:
            return indices_new

        sameClassWindowIndex = np.arange(self.indices.shape[0])[indices_new]
        
        return sameClassWindowIndex, self.SCS[self.indices[sameClassWindowIndex]]
    
    def getChangePointIndex(self):
        diff = np.diff(self.SCS)
        ids = np.nonzero(diff)[0]

        temp_indices = np.zeros_like(self.SCS, dtype=np.bool_)
        temp_indices[ids] = True

        indices_new = temp_indices[self.indices]
        changePointIndex = np.arange(self.indices.shape[0])[indices_new]

        return changePointIndex

    def normalizeSTS(self, mode):
        self.mean = np.expand_dims(self.STS.mean(1), 1)
        self.std = np.expand_dims(np.std(self.STS, axis=1), 1)

        self.STS = (self.STS - self.mean) / self.std

class StreamingTimeSeries(STSDataset):

    def __init__(self,
            STS: np.ndarray,
            SCS: np.ndarray,
            wsize: int = 10,
            wstride: int = 1,
            normalize: bool = True,
            label_mapping: np.ndarray = None
            ) -> None:
        super().__init__(wsize=wsize, wstride=wstride)

        self.STS = STS
        self.SCS = SCS

        self.splits = np.array([0, SCS.shape[0]])

        # process ds
        self.label_mapping = label_mapping
        if not self.label_mapping is None:
            self.SCS = self.label_mapping[self.SCS]

        self.indices = np.arange(self.SCS.shape[0])
        for i in range(wsize * wstride):
            self.indices[self.splits[:-1] + i] = 0
        self.indices = self.indices[np.nonzero(self.indices)]

        if normalize:
            self.normalizeSTS("normal")


def return_indices_train(x, subjects, subject_splits):
    out = np.ones_like(x, dtype=bool)
    for s in subjects:
        out = out & ((x<subject_splits[s]) | (x>subject_splits[s+1]))
    return out

def return_indices_test(x, subjects, subject_splits):
    out = np.zeros_like(x, dtype=bool)
    for s in subjects:
        out = out | ((x>subject_splits[s]) & (x<subject_splits[s+1]))
    return out

# series splitting functions

def split_by_test_subject(sts, test_subject, n_val_subjects, seed=42):
    if hasattr(sts, "subject_indices"):
        subject_splits = sts.subject_indices
    else:
        subject_splits = list(sts.splits)

    rng = np.random.default_rng(seed)

    val_subject_indices = np.arange(len(subject_splits) - 1)
    val_subjects_selected = list(rng.choice(val_subject_indices, n_val_subjects, replace=False))
    
    if not isinstance(test_subject, list):
        test_subject = [test_subject]

    for s in test_subject:
        if s > len(subject_splits) - 1:
            raise Exception(f"No subject with index {s}")

    return {
        "train": lambda x: return_indices_train(x, subjects=test_subject + val_subjects_selected, subject_splits=subject_splits),
        "val": lambda x: return_indices_test(x, subjects=val_subjects_selected, subject_splits=subject_splits),
        "test": lambda x: return_indices_test(x, subjects=test_subject, subject_splits=subject_splits),
    }