import os
import hashlib

from typing import List, Tuple, Dict
from collections import namedtuple

import numpy as np
from scipy import stats

from data_2.stsdataset import STSDataset
from data_2.df.patterns import get_patterns
from transforms.odtw import compute_oDTW, compute_oDTW_channel

# pylint: disable=invalid-name too-many-instance-attributes too-many-arguments

PatternConf = namedtuple("PatternConf",
    ["pattern_type", "pattern_size", "rho", "cached", "compute_n"])

class DFDataset(STSDataset):
    def __init__(self,
            patterns: PatternConf = PatternConf(None, None, .1, False, 100),
            data: List[List[Tuple[np.ndarray, np.ndarray]]] = None,
            wsize: int = 10,
            wstride: int = 1,
            minmax: Tuple[np.ndarray, np.ndarray] = None,
            label_mapping: np.ndarray = np.arange(256, dtype=np.int64),
            label_mode: int = 0,
            feature_group: List[np.ndarray] = None,
            triplets: bool = False,
            computed_patterns: np.ndarray = None
            ) -> None:
        '''
            patterns: shape (n_shapes, channels, pattern_size)
        '''
        super().__init__(
            data, wsize, wstride, minmax, label_mapping, label_mode, feature_group, triplets)

        self.cached = patterns.cached

        if computed_patterns is None:
            self.patterns = get_patterns(
                patterns.pattern_type, patterns.pattern_size,
                patterns.compute_n, self.stream, self.labels)
        else:
            self.patterns = computed_patterns
        self.rho = patterns.rho

        self.DM = []

        patt_hash = hashlib.sha1(self.patterns.data)
        cache_id = patt_hash.hexdigest()
        self.cache_dir = os.path.join(os.getcwd(), "cache_df_" + cache_id)
        print("hash of patterns:", cache_id)

        for s in range(self.splits.shape[0] - 1):
            DM = self._compute_dm(self.patterns, self.splits[s:s+2])
            self.DM.append(DM)

        self.n_patterns = self.DM[0].shape[1]

    def _compute_dm(self, pattern, split):
        if len(pattern.shape) == 3:
            DM = compute_oDTW(self.stream[split[0]:split[1], :].T, pattern, rho=self.rho)
        elif len(pattern.shape) == 2:
            DM = compute_oDTW_channel(
                self.stream[split[0]:split[1], :].T, pattern, rho=self.rho)

        # put time dimension in the first dimension
        DM = np.ascontiguousarray(np.transpose(DM, (2, 0, 1)))
        # therefore, DM has dimensions (n, num_frames, patt_len)

        return DM

    def triplet(self, c: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        # get the close and far samples for triplet/contrastive learning

        close_id = np.random.choice(self.per_class[c], 1).item()
        far_cl = (np.random.choice(len(self.per_class) - 1) + c + 1) % len(self.per_class)

        # el sampleo del negativo (far_cl) está bien, cualquier carencia es debido al modelo
        far_id = np.random.choice(self.per_class[far_cl], 1).item()

        close_split = self.id_to_split[close_id]
        close_first = close_id - self.wsize*self.wstride + self.wstride - self.splits[close_split]
        close_last = close_id + 1 - self.splits[close_split]

        far_split = self.id_to_split[far_id]
        far_first = far_id - self.wsize*self.wstride + self.wstride - self.splits[far_split]
        far_last = far_id + 1 - self.splits[close_split]

        close_df = self.DM[close_split][close_first:close_last:self.wstride]
        far_df = self.DM[far_split][far_first:far_last:self.wstride]

        return {
            "triplet": (
            self.stream[close_first:close_last:self.wstride].T,
            self.stream[far_first:far_last:self.wstride].T
            ),
            "df_triplet": (
                close_df.transpose((1, 2, 0)),
                far_df.transpose((1, 2, 0))
            )
        }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        pos = self.indices[index]

        first = pos - self.wsize*self.wstride + self.wstride
        last = pos + 1

        scs = self.labels[first:last:self.wstride]
        if self.label_mode > 1:
            c = stats.mode(scs[-self.label_mode:])
        else:
            c = scs[-1]

        split = self.id_to_split[pos]
        dm_first = first - self.splits[split]
        dm_last = last - self.splits[split]
        df = self.DM[split][dm_first:dm_last:self.wstride].transpose((1, 2, 0))

        return {
            "series": self.stream[first:last:self.wstride].T,
            "scs": scs,
            "label": c,
            "df": df,
            **(self.triplet(c) if self.get_triplets else {"triplet": 0, "df_triplet": 0})
        }
