# pylint: disable=too-many-locals invalid-name

import numpy as np

from aeon.distances import pairwise_distance

# Methods to obtain patterns

def compute_medoids(
        X: np.ndarray, Y: np.ndarray,
        meds_per_class: int = 1, metric: str = 'dtw',
    ) -> tuple[np.ndarray, np.ndarray]:

    """ Computes 'meds_per_class' medoids of each class in the dataset. """

    # Check the distance type
    suported_metrics = ['euclidean', 'squared',
        'dtw', 'ddtw', 'wdtw', 'wddtw', 'lcss', 'edr', 'erp', 'msm']
    if metric not in suported_metrics:
        raise ValueError(f"The distance type must be one of {suported_metrics}.")

    # grab the classes
    sdim, slen = X.shape[1], X.shape[2]
    classes = np.unique(Y)

    # Initialize the arrays
    meds = np.empty((len(np.unique(Y)), meds_per_class, sdim, slen), dtype=float)
    meds_idx = np.empty((len(np.unique(Y)), meds_per_class), dtype=int)

    # Find the medoids for each class
    for i, y in enumerate(classes):
        index = np.argwhere(Y == y)[:,0]
        X_y = X[index,:,:]
        dm = pairwise_distance(X_y, metric=metric)
        scores = dm.sum(axis=0)
        meds_idx_y = np.argpartition(scores, meds_per_class)[:meds_per_class]
        meds[i,:,:,:] = X_y[meds_idx_y]
        meds_idx[i,:] = index[meds_idx_y]

    # Return the medoids and their indices
    return meds, meds_idx

def sts_medoids(
        sts, scs, n = 100, pattern_size = -1, meds_per_class = 1, random_seed: int = 42):
    np.random.seed(random_seed)

    diff = np.diff(scs)
    ids = np.concatenate(([0], np.nonzero(diff)[0], [sts.shape[0]]))

    temp_indices = np.zeros_like(scs, dtype=np.bool_)

    offset = pattern_size + 1
    for i in range(ids.shape[0]-1):
        if ids[i+1] - ids[i] >= offset:
            temp_indices[(ids[i] + offset):(ids[i+1]+1)] = True

    indices = np.nonzero(temp_indices)[0]
    valid = indices[scs[indices] != 100]

    selected_w = []
    selected_c = []

    for c in np.unique(scs[valid]):
        # get the random windows for the class c

        rw = np.random.choice(valid[scs[valid] == c], n)

        for i in rw:
            selected_w.append(sts[i-pattern_size+1:i+1].T)

        selected_c.append(np.full(n, c, np.int32))

    selected_w = np.stack(selected_w, axis=0) # (n, dims, len)
    meds, _ = compute_medoids(
        selected_w, np.concatenate(selected_c, axis=0), meds_per_class=meds_per_class)

    return meds.reshape((meds.shape[0]*meds.shape[1], meds.shape[2], meds.shape[3]))


# fft-based pattern frequency extraction

def process_fft_frequencies(STS, SCS, frequency_values):
    # STS of shape (n, n_classes)

    class_changes = list(np.nonzero(np.diff(SCS))[0])
    if class_changes[0] != 0:
        class_changes = [0] + class_changes

    magnitudes = {} # a dict for each class
    classes = np.unique(SCS)
    for c in classes:
        # magnitudes[class] of shape (n_classes, freq_values)
        magnitudes[c] = np.zeros((STS.shape[1], frequency_values.shape[0]), dtype=np.complex128)
    class_counts = np.zeros(256)

    for i in range(len(class_changes)-1):
        current_class = SCS[class_changes[i]+1]

        series_part = STS[(class_changes[i]+1):(class_changes[i+1]+1)].T
        fft_size = series_part.shape[1]
        if fft_size<3:
            continue

        fft_short = np.fft.fft(series_part, axis=-1, n=fft_size)[:, :(fft_size//2)]
        fft_freq = np.fft.fftfreq(fft_size)[:(fft_size//2)]

        for c in range(fft_short.shape[0]):
            freq_val = np.interp(frequency_values, fft_freq, fft_short[c, :])
            div = class_counts[current_class]+1
            magnitudes[current_class][c, :] *= class_counts[current_class]/div
            magnitudes[current_class][c, :] += freq_val/div

        class_counts[current_class] += 1

    return magnitudes
