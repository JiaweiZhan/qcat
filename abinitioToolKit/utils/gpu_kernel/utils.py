USE_GPU = False
try:
    import cupy as np
    USE_GPU = True
except ImportError:
    import numpy as np
import numpy as numpy

def gaussian3d(unit_cell,      # [3, 3]
               r1: numpy.ndarray, # [ngrid, 3] in relative scale
               r2: numpy.ndarray, # [nefield // 2, nmlwf, 3] in relative scale
               spread: numpy.ndarray, # [nefield //2, nmlwf, 1] in bohr unit
               dspl_norm: numpy.ndarray, # [nefield // 2, nmlwf, 1] in bohr unit
               ):
    '''
    return shape: [nefield // 2, ngrid]
    FIXME: this function is not optimized, GPU acceleration per grid is possible.
    '''
    if not USE_GPU:
        return gaussian3d_helper(unit_cell, r1, r2, spread, dspl_norm)
    else:
        unit_cell = np.asarray(unit_cell)
        r2 = np.asarray(r2)
        spread = np.asarray(spread)
        dspl_norm = np.asarray(dspl_norm)

        free_memory, _ = np.cuda.runtime.memGetInfo()
        ngrid = r1.shape[0]
        nefield, nmlwf, _ = r2.shape
        nr1 = free_memory // (r1.itemsize * nefield * nmlwf * 3 * 2)
        lenr1 = (ngrid - 1) // nr1 + 1
        result = []
        for i in range(lenr1):
            low = i * nr1
            high = min(ngrid, (i + 1) * nr1)
            r1_gpu = np.asarray(r1[low : high])
            result.append(gaussian3d_helper(unit_cell, r1_gpu, r2, spread, dspl_norm))
        return np.concatenate(result, axis=-1).get()

def gaussian3d_helper(unit_cell,      # [3, 3]
                      r1: np.ndarray, # [ngrid, 3] in relative scale
                      r2: np.ndarray, # [nefield // 2, nmlwf, 3] in relative scale
                      spread: np.ndarray, # [nefield //2, nmlwf, 1] in bohr unit
                      dspl_norm: np.ndarray, # [nefield // 2, nmlwf, 1] in bohr unit
                      ):
    '''
    return shape: [nefield // 2, ngrid]
    FIXME: this function is not optimized, GPU acceleration per grid is possible.
    '''
    diff = r1[None, :, None, :] - r2[:, None, :, :] # [nefield // 2, ngrid, nmlwf, 3]
    diff = (diff + 0.5) % 1 - 0.5
    diff_norm = np.linalg.norm(diff @ unit_cell, axis=-1) # [nefield // 2, ngrid, nmlwf]
    dspl_norm = dspl_norm.transpose(0, 2, 1)                   # [nefield // 2, 1, nmlwf]
    spread = spread.transpose(0, 2, 1)                   # [nefield // 2, 1, nmlwf]
    return np.sum(dspl_norm * np.exp(-0.5 * diff_norm ** 2 / spread ** 2) / ((2 * np.pi * spread ** 2) ** 1.5), axis=-1) # [nefield // 2, ngrid]
