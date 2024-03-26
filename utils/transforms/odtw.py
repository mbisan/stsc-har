from numba import jit, prange
import numpy as np

# pylint: disable=invalid-name not-an-iterable

@jit(nopython=True, parallel=True)
def fill_first_line(DM: np.array, w: float) -> None:
    for p in prange(DM.shape[0]):
        for j in range(1, DM.shape[2]):
            DM[p, 0, j] += w*DM[p, 0, j-1]

@jit(nopython=True, parallel=True)
def fill_dtw(DM: np.ndarray, w: float) -> None:
    for p in prange(DM.shape[0]):
        for i in range(1, DM.shape[1]):
            for j in range(1, DM.shape[2]):
                temp = w * np.minimum(DM[p, i, j-1], DM[p, i-1, j-1])
                DM[p, i, j] += np.minimum(temp, DM[p, i-1, j])

@jit(nopython=True, parallel=True)
def compute_pointwise_ED2(DM: np.ndarray, STS: np.ndarray, patts: np.ndarray) -> None:
    for p in prange(DM.shape[0]):
        for i in range(DM.shape[1]):
            for j in range(DM.shape[2]):
                DM[p, i, j] = np.sum(np.square(STS[:, j] - patts[p, :, i]))

def compute_oDTW(
        STS: np.ndarray,
        patts: np.ndarray,
        rho: float,
        ) -> np.ndarray:
    '''
        STS has shape (c, n)
        patts has shape (n_patts, c, m)
        rho: weight of the o-DTW for the (-m)-th element in the STS

        returns: DM of shape (n_patts, m, n)
    '''

    assert STS.shape[0] == patts.shape[1]

    lpatts: int = patts.shape[2]
    w: np.float32 = rho**(1/lpatts)

    DM = np.empty((patts.shape[0], patts.shape[-1], STS.shape[1]), dtype=np.float32)

    # Compute point-wise distances
    compute_pointwise_ED2(DM, STS, patts)

    # incorrect computation of this value
    # np.cumsum(DM[:,0,:], axis=1, out=DM[:,0,:])

    # correct
    fill_first_line(DM, w)

    np.cumsum(DM[:,:,0], axis=1, out=DM[:,:,0])

    fill_dtw(DM, w)

    # Return the DM
    return np.sqrt(DM)

@jit(nopython=True, parallel=True)
def fill_first_line_channel(DM: np.array, w: float) -> None:
    for c in prange(DM.shape[1]):
        for p in prange(DM.shape[0]):
            for j in range(1, DM.shape[3]):
                DM[p, c, 0, j] += w*DM[p, c, 0, j-1]

# Channel-wise computation of the DM
@jit(nopython=True, parallel=True)
def fill_dtw_channel(DM: np.ndarray, w: float) -> None:
    for c in prange(DM.shape[1]):
        for p in prange(DM.shape[0]):
            for i in range(1, DM.shape[2]):
                for j in range(1, DM.shape[3]):
                    temp = w * np.minimum(DM[p, c, i, j-1], DM[p, c, i-1, j-1])
                    DM[p, c, i, j] += np.minimum(temp, DM[p, c, i-1, j])

@jit(nopython=True, parallel=True)
def compute_pointwise_ED2_channel(DM: np.ndarray, STS: np.ndarray, patts: np.ndarray) -> None:
    for c in prange(DM.shape[1]):
        for p in prange(DM.shape[0]):
            for i in range(DM.shape[2]):
                for j in range(DM.shape[3]):
                    DM[p, c, i, j] = np.square(STS[c, j] - patts[p, i])

def compute_oDTW_channel(
        STS: np.ndarray,
        patts: np.ndarray,
        rho: float,
        ) -> np.ndarray:
    '''
        STS has shape (c, n)
        patts has shape (n_patts, m)
        rho: weight of the o-DTW for the (-m)-th element in the STS

        returns: DM of shape (n_patts*c, m, n)
        i.e. each pattern is compared to each channel of the STS
    '''

    lpatts: int = patts.shape[1]
    w: np.float32 = rho**(1/lpatts)

    DM = np.empty((patts.shape[0], STS.shape[0], patts.shape[-1], STS.shape[1]), dtype=np.float32)

    # Compute point-wise distances
    compute_pointwise_ED2_channel(DM, STS, patts)

    # incorrect computation of this value
    # np.cumsum(DM[:,:,0,:], axis=2, out=DM[:,:,0,:])

    # correct
    fill_first_line_channel(DM, w)

    np.cumsum(DM[:,:,:,0], axis=2, out=DM[:,:,:,0])

    fill_dtw_channel(DM, w)

    # Return the DM
    return np.sqrt(DM.reshape((patts.shape[0]*STS.shape[0], patts.shape[-1], STS.shape[1])))
