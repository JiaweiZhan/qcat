import numpy as np

def gaussian3d_cpu(unit_cell,      # [3, 3]
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
