try:

    from cupyx import jit
    import cupy
    import numpy as np

    @jit.rawkernel()
    def gaussian3d_gpu_kernel(unit_cell,
                            r1,
                            r2,
                            spread,
                            dspl_norm,
                            result,
                            ngrid,
                            nefield,
                            nmlwf,
                            ):
        bid = jit.blockIdx.x
        tid = jit.threadIdx.x
        idx = bid * jit.blockDim.x + tid
        offset = jit.blockDim.x * jit.gridDim.x
        pi2 = cupy.pi * 2

        for igrid in range(idx, ngrid, offset):
            for iefield in range(nefield):
                for imlwf in range(nmlwf):
                    lspread2 = spread[iefield, imlwf, 0] ** 2
                    ldspl_norm = dspl_norm[iefield, imlwf, 0]
                    ddx = r1[igrid, 0] - r2[iefield, imlwf, 0]
                    ddx = (ddx + 0.5) % 1 - 0.5
                    ddy = r1[igrid, 1] - r2[iefield, imlwf, 1]
                    ddy = (ddy + 0.5) % 1 - 0.5
                    ddz = r1[igrid, 2] - r2[iefield, imlwf, 2]
                    ddz = (ddz + 0.5) % 1 - 0.5
                    ddx_temp = ddx * unit_cell[0, 0] + ddy * unit_cell[1, 0] + ddz * unit_cell[2, 0]
                    ddy_temp = ddx * unit_cell[0, 1] + ddy * unit_cell[1, 1] + ddz * unit_cell[2, 1]
                    ddz_temp = ddx * unit_cell[0, 2] + ddy * unit_cell[1, 2] + ddz * unit_cell[2, 2]
                    diff_norm = (ddx_temp ** 2 + ddy_temp ** 2 + ddz_temp ** 2) ** 0.5
                    result[iefield, igrid] += ldspl_norm * cupy.exp(-0.5 * diff_norm ** 2 / lspread2) / ((pi2 * lspread2) ** 1.5)

    def gaussian3d_gpu(unit_cell,      # [3, 3]
                    r1: np.ndarray, # [ngrid, 3] in relative scale
                    r2: np.ndarray, # [nefield // 2, nmlwf, 3] in relative scale
                    spread: np.ndarray, # [nefield //2, nmlwf, 1] in bohr unit
                    dspl_norm: np.ndarray, # [nefield // 2, nmlwf, 1] in bohr unit
                    ):
        ngrid = r1.shape[0]
        nefield, nmlwf, _ = r2.shape
        nthread = 512
        nblock = (ngrid + nthread - 1) // nthread
        unit_cell = cupy.asarray(unit_cell)
        r1_gpu = cupy.asarray(r1)
        r2_gpu = cupy.asarray(r2)
        spread_gpu = cupy.asarray(spread)
        dspl_norm_gpu = cupy.asarray(dspl_norm)
        result_gpu = cupy.zeros((nefield, ngrid), dtype=r1.dtype)
        gaussian3d_gpu_kernel[nblock, nthread](unit_cell, r1_gpu, r2_gpu, spread_gpu, dspl_norm_gpu, result_gpu, ngrid, nefield, nmlwf)
        return result_gpu.get()

except ImportError:
    pass
