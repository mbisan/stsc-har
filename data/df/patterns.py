import numpy as np
from data.df.pattern_extract import (
    sts_medoids, process_fft_frequencies)

def get_patterns(pattern_type, pattern_size, compute_n, sts, scs):
    if pattern_type == "med":
        print("Computing medoids...")
        meds = sts_medoids(
            sts, scs, pattern_size=pattern_size, meds_per_class=1, n=compute_n)
    if pattern_type == "med_mean":
        print("Computing medoids...")
        meds = sts_medoids(
            sts, scs, pattern_size=pattern_size, meds_per_class=1, n=compute_n)
        meds = np.mean(meds, axis=0, keepdims=True)
    elif pattern_type == "noise":
        print("Using gaussian noise...")
        meds = np.random.randn(np.sum(np.unique(scs)!=100), sts.shape[0], pattern_size)
        print(meds.shape)
    elif pattern_type == "noise_1":
        print("Using gaussian noise...")
        meds = np.random.randn(1, sts.shape[0], pattern_size)
        print(meds.shape)
    elif pattern_type == "noise_c":
        print("Using gaussian noise...")
        meds = np.random.randn(3, pattern_size)
        print(meds.shape)
    elif pattern_type == "noise_c_1":
        print("Using gaussian noise...")
        meds = np.random.randn(1, pattern_size)
        print(meds.shape)
    elif pattern_type == "syn":
        print("Using synthetic shapes...")
        meds = np.empty((3, pattern_size))
        meds[0,:] = np.linspace(-1, 1, pattern_size)
        meds[1,:] = np.linspace(1, -1, pattern_size)
        meds[2,:] = 0
    elif pattern_type == "syn_1":
        print("Using up shape...")
        meds = np.empty((1, pattern_size))
        meds[0,:] = np.linspace(-1, 1, pattern_size)
    elif pattern_type == "syn_2":
        print("Using up down shape...")
        meds = np.empty((2, pattern_size))
        meds[0,:] = np.linspace(-1, 1, pattern_size)
        meds[1,:] = np.linspace(1, -1, pattern_size)
    elif pattern_type == "syn_g":
        print("Using synthetic shapes with gaussian noise...")
        meds = np.empty((3, pattern_size))
        meds[0,:] = np.linspace(-1, 1, pattern_size) + 0.2 * np.random.randn(pattern_size)
        # sigma = 0.2*0.2 = 0.04 (std)
        meds[1,:] = np.linspace(1, -1, pattern_size) + 0.2 * np.random.randn(pattern_size)
        meds[2,:] = 0.1 * np.random.randn(pattern_size)
    elif pattern_type == "f1":
        meds = np.empty((1, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size) # one cycle
    elif pattern_type == "f2":
        meds = np.empty((1, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size) # two cycle
    elif pattern_type == "f4":
        meds = np.empty((1, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size) # four cycle
    elif pattern_type == "f12":
        meds = np.empty((2, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size)
        meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size)
    elif pattern_type == "f14":
        meds = np.empty((2, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size)
        meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size)
    elif pattern_type == "f24":
        meds = np.empty((2, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size)
        meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size)
    elif pattern_type == "f124":
        meds = np.empty((3, pattern_size))
        meds[0,:] = np.sin(2*np.pi* np.arange(pattern_size) * 1/pattern_size)
        meds[1,:] = np.sin(2*np.pi* np.arange(pattern_size) * 2/pattern_size)
        meds[2,:] = np.sin(2*np.pi* np.arange(pattern_size) * 4/pattern_size)
    elif pattern_type == "fftcoef":
        print("Using fft coefficients for the pattern...")
        pattern_freq = np.fft.fftfreq(pattern_size)[:pattern_size//2]
        fft_coef = process_fft_frequencies(sts, scs, pattern_freq)

        if 100 in fft_coef:
            del fft_coef[100] # remove the ignore label

        meds = np.zeros((len(fft_coef.keys()), sts.shape[1], pattern_size))
        # num_classes, channel, pattern_size

        for i, coef in enumerate(fft_coef.values()):
            for c in range(meds.shape[1]):
                for j, m in enumerate(pattern_freq):
                    meds[i, c, :] += coef[c, j].real * np.sin(2*np.pi* m * np.arange(pattern_size))
                    meds[i, c, :] += coef[c, j].imag * np.cos(2*np.pi* m * np.arange(pattern_size))
    elif pattern_type == "fftvar":
        print("Using fft coefficient variances across classes for the pattern...")
        pattern_freq = np.fft.fftfreq(pattern_size)[:pattern_size//2]
        fft_coef = process_fft_frequencies(sts, scs, pattern_freq)

        if 100 in fft_coef:
            del fft_coef[100] # remove the ignore label

        # compute variance across classes
        fft_coef_all = np.zeros(
            (len(fft_coef.keys()), sts.shape[1], pattern_freq.shape[0]), dtype=np.complex128)
        for i, v in enumerate(fft_coef.values()):
            fft_coef_all[i, :, :] = v

        fft_coef_mean = np.mean(fft_coef_all, axis=(0, 1))

        fft_var = np.var(np.abs(fft_coef_all), axis=(0, 1)) # shape of pattern_freq
        fft_freq_ordered = pattern_freq[np.argsort(fft_var)] # from lower to higher variance
        fft_coef_mean_sorted = fft_coef_mean[np.argsort(fft_var)] # from lower to higher variance

        num_waves = 3
        meds = np.zeros((num_waves, pattern_size)) # num_classes, channel, pattern_size
        for i in range(num_waves):
            meds[i, :] += fft_coef_mean_sorted[-i].real * \
                np.sin(2*np.pi* fft_freq_ordered[-i] * np.arange(pattern_size))
            meds[i, :] += fft_coef_mean_sorted[-i].imag * \
                np.cos(2*np.pi* fft_freq_ordered[-i] * np.arange(pattern_size))

    return meds
