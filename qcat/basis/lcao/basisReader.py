import numpy as np
from typing import Tuple
from scipy.special import sph_harm

class lcaoReader(object):
    def __init__(self,
                 fname: str):
        self.fname = fname
        self.element_type = ""
        self.basis_set = {}
        self.mesh = 1
        self.dr = 0.01
        self.read_lcao()

    def read_lcao(self):
        nbasis = 0
        with open(self.fname, 'r') as file_obj:
            file_obj.readline()
            self.element_type = file_obj.readline().strip().split()[-1].strip()

            for _ in range(2):
                file_obj.readline()
            lmax = int(file_obj.readline().strip().split()[-1].strip())

            for _ in range(lmax + 1):
                nbasis += int(file_obj.readline().strip().split()[-1].strip())

            for _ in range(3):
                file_obj.readline()

            self.mesh = int(file_obj.readline().strip().split()[-1].strip())
            self.dr = float(file_obj.readline().strip().split()[-1].strip())

            for _ in range(nbasis):
                file_obj.readline()
                line = file_obj.readline().strip().split()
                l = int(line[1])
                n = int(line[2])
                basis = np.fromfile(file_obj, count=self.mesh, sep=' ')
                self.basis_set[(n,l)] = basis

    def eval_ao(self,
                cell: np.ndarray,  # [3, 3] in bohr unit, [i, 1:3] is the i-th lattice vector
                fftw: Tuple,       # [3] fftw shape
                ):
        grid = np.mgrid[0:fftw[0], 0:fftw[1], 0:fftw[2]]
        grid = grid.astype(np.float64)
        grid[0] /= fftw[0]
        grid[1] /= fftw[1]
        grid[2] /= fftw[2]

        grid = np.transpose(grid, (1, 2, 3, 0))  # [x, y, z, 3]
        grid = (grid + 0.5) % 1 - 0.5
        grid = grid.reshape(-1, 3) # [ngrid, 3]
        grid = grid @ cell         # [ngrid, 3]
        r_norm = np.linalg.norm(grid, axis=-1)
        theta = np.arccos(grid[1:, 2] / r_norm[1:])
        theta = np.pad(theta, (1, 0), mode='constant', constant_values=0.0)
        phi = np.arctan2(grid[1:, 1], grid[1:, 0])
        phi = np.pad(phi, (1, 0), mode='constant', constant_values=0.0)
        r_norm_idx = np.round(r_norm / self.dr).astype(np.int32) # [ngrid]
        r_norm_idx = np.clip(r_norm_idx, 0, self.mesh - 1)

        results = []
        basis_name = []
        for key, value in self.basis_set.items():
            (n, l) = key
            for m in range(-l, l + 1):
                if m == 0:
                    angular_part = sph_harm(m, l, phi, theta).real
                elif m < 0:
                    angular_part = complex(0, 1) * np.sqrt(0.5) * (sph_harm(m, l, phi, theta) - (-1) ** (-m) * sph_harm(-m, l, phi, theta))
                    angular_part = angular_part.real
                else:
                    angular_part = np.sqrt(0.5) * (sph_harm(-m, l, phi, theta) + (-1) ** m * sph_harm(m, l, phi, theta))
                    angular_part = angular_part.real

                basis = value[r_norm_idx] * angular_part
                results.append(basis.reshape(*fftw))
                basis_name.append((n, l, m))
        return basis_name, np.stack(results)


    def __str__(self):
        return f"Element type: {self.element_type}\n" + \
               f"Basis set: {self.basis_set}\n" + \
               f"Mesh: {self.mesh}\n" + \
               f"dr: {self.dr}"
