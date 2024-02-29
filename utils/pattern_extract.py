# pylint: disable=too-many-locals invalid-name

import numpy as np

from aeon.distances import pairwise_distance
from tslearn.clustering import TimeSeriesKMeans

from data.base import STSDataset

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
        dataset: STSDataset, n = 100, pattern_size = -1, meds_per_class = 1, random_seed: int = 42):
    np.random.seed(random_seed)

    window_id, window_lb = dataset.getSameClassWindowIndex()

    selected_w = []
    selected_c = []

    for c in np.unique(window_lb):
        # get the random windows for the class c

        rw = np.random.choice(window_id[window_lb == c].reshape(-1), n)

        ts, _ = dataset.sliceFromArrayOfIndices(rw)

        selected_w.append(ts)
        selected_c.append(np.full(n, c, np.int32))

    selected_w = np.concatenate(selected_w) # (n, dims, len)
    if pattern_size>0:
        selected_w = selected_w[:,:,-pattern_size:]
    meds, _ = compute_medoids(
        selected_w, np.concatenate(selected_c, axis=0), meds_per_class=meds_per_class)

    return meds.reshape((meds.shape[0]*meds.shape[1], meds.shape[2], meds.shape[3]))


def sts_barycenter(dataset: STSDataset, n: int = 100, random_seed: int = 42):
    np.random.seed(random_seed)

    window_id, window_lb = dataset.getSameClassWindowIndex()
    selected = np.empty((np.unique(window_lb).shape[0], dataset.STS.shape[0], dataset.wsize))

    for i, c in enumerate(np.unique(window_lb)):
        # get the random windows for the class c

        rw = np.random.choice(window_id[window_lb == c].reshape(-1), n)

        ts, _ = dataset.sliceFromArrayOfIndices(rw)

        km = TimeSeriesKMeans(n_clusters=1, verbose=True, random_state=1, metric="dtw", n_jobs=-1)
        km.fit(np.transpose(ts, (0, 2, 1)))

        selected[i] = km.cluster_centers_[0].T

    return selected


# fft-based pattern frequency extraction

def process_fft(STS, SCS):
    class_changes = [0] + list(np.nonzero(np.diff(SCS))[0])

    magnitudes = {} # a dict for each class
    classes = np.unique(SCS)
    for c in classes:
        magnitudes[c] = [{} for i in range(STS.shape[0])] # a dict for each channel

    for i in range(len(class_changes)-1):
        current_class = SCS[class_changes[i]+1].item()

        series_part = STS[:, (class_changes[i]+1):(class_changes[i+1]+1)]
        fft_size = 2**int(np.log2(series_part.shape[1]))
        fft_short = np.fft.fft(series_part, axis=-1, n=fft_size)
        # highest frequencies for signals of sampling rate 50 is 25
        fft_freq = np.fft.fftfreq(fft_size)

        for c in range(fft_short.shape[0]):
            for j in range(fft_short.shape[1]):
                tmp = magnitudes[current_class][c].get(fft_freq[j], 0) + np.abs(fft_short[c, j])
                magnitudes[current_class][c][fft_freq[j]] = tmp

    return magnitudes

def get_predominant_frequency(fft_mag, mode="per_class"):
    classes_list = list(filter(lambda x: x!=100, fft_mag.keys()))
    num_classes = len(classes_list)

    if mode=="per_class":
        # (n, c) we get a predominant frequency per channel, per class
        out = np.zeros((num_classes, len(fft_mag[0])))

        for i, c in enumerate(classes_list):
            for j, channel_result in enumerate(fft_mag[c]):
                sorted_fr = list(
                    filter(lambda x: x[0]>0, sorted(channel_result.items(), key=lambda x:x[1])))
                out[i, j] = sorted_fr[-1][0]

        return out

    # elif mode=="per_channel": # get the frequencies with most importance in the fft transform
    out = {} # frequencies magnitude total

    for i, c in enumerate(classes_list):
        for j, channel_result in enumerate(fft_mag[c]):
            for f in channel_result.keys():
                out[f] = out.get(f, 0) + channel_result[f]

    frequencies_ordered = np.array(list(
        filter(lambda x: x[0]>0, sorted(out.items(), key=lambda x:x[1], reverse=True))
    ))

    return frequencies_ordered

def process_fft_frequencies(STS, SCS, frequency_values):
    class_changes = list(np.nonzero(np.diff(SCS))[0])
    if class_changes[0] != 0:
        class_changes = [0] + class_changes

    magnitudes = {} # a dict for each class
    classes = np.unique(SCS)
    for c in classes:
        magnitudes[c] = np.zeros((STS.shape[0], frequency_values.shape[0]), dtype=np.complex128)
    class_counts = np.zeros(256)

    for i in range(len(class_changes)-1):
        current_class = SCS[class_changes[i]+1]

        series_part = STS[:, (class_changes[i]+1):(class_changes[i+1]+1)]
        series_part = (
            series_part - np.mean(series_part, axis=1, keepdims=True)
            ) / (np.std(series_part, axis=1, keepdims=True) + 1e-6)
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
