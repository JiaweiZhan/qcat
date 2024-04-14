import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gaussian3d(unit_cell,      # [3, 3]
               r1,             # [ngrid, 3] in relative scale
               r2,             # [nefield // 2, nmlwf, 3] in relative scale
               spread,         # [nefield //2, nmlwf, 1] in bohr unit
               dspl_norm,      # [nefield // 2, nmlwf, 1] in bohr unit
               ):
    '''
    return shape: [nefield // 2, ngrid]
    FIXME: this function is not optimized, GPU acceleration per grid is possible.
    '''
    with torch.no_grad():
        unit_cell = torch.as_tensor(unit_cell, device=device)
        r2 = torch.as_tensor(r2, device=device)
        spread = torch.as_tensor(spread, device=device)
        dspl_norm = torch.as_tensor(dspl_norm, device=device)

        if not torch.cuda.is_available():
            r1 = torch.as_tensor(r1, device=device)
            return gaussian3d_helper(unit_cell, r1, r2, spread, dspl_norm)
        else:
            free_memory, _ = torch.cuda.mem_get_info()
            ngrid = r1.shape[0]
            nefield, nmlwf, _ = r2.shape
            nr1 = free_memory // (r1.itemsize * nefield * nmlwf * 3 * 10)
            lenr1 = (ngrid - 1) // nr1 + 1
            result = []
            for i in range(lenr1):
                low = i * nr1
                high = min(ngrid, (i + 1) * nr1)
                r1_gpu = torch.as_tensor(r1[low : high], device=device)
                result.append(gaussian3d_helper(unit_cell, r1_gpu, r2, spread, dspl_norm))
            return torch.cat(result, dim=-1).cpu().numpy()

def gaussian3d_helper(unit_cell: torch.Tensor,      # [3, 3]
                      r1: torch.Tensor,             # [ngrid, 3] in relative scale
                      r2: torch.Tensor,             # [nefield // 2, nmlwf, 3] in relative scale
                      spread: torch.Tensor,         # [nefield //2, nmlwf, 1] in bohr unit
                      dspl_norm: torch.Tensor,      # [nefield // 2, nmlwf, 1] in bohr unit
                      ):
    '''
    return shape: [nefield // 2, ngrid]
    FIXME: this function is not optimized, GPU acceleration per grid is possible.
    '''
    diff = r1[None, :, None, :] - r2[:, None, :, :] # [nefield // 2, ngrid, nmlwf, 3]
    diff = (diff + 0.5) % 1 - 0.5
    diff_norm = torch.linalg.norm(diff @ unit_cell, dim=-1) # [nefield // 2, ngrid, nmlwf]
    dspl_norm = dspl_norm.permute(0, 2, 1)                   # [nefield // 2, 1, nmlwf]
    spread = spread.permute(0, 2, 1)                   # [nefield // 2, 1, nmlwf]
    return torch.sum(dspl_norm * torch.exp(-0.5 * diff_norm ** 2 / spread ** 2) / ((2 * math.pi * spread ** 2) ** 1.5), dim=-1) # [nefield // 2, ngrid]
