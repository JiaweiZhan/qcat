from pyscf import pbc
from typing import List, Tuple
import os
import numpy as np
from e3nn.o3 import spherical_harmonics
import torch
from pyscf.lib import param

from .basisReader import lcaoReader

class lcaoGenerator(object):
    def __init__(self,
                 cell: pbc.gto.Cell,
                 basis_fname: List,
                 fftw: Tuple,):
        self.cell_ = cell
        self.basis_fname_ = basis_fname
        self.fft_ = fftw
        self.basis_ = []
        self.species_ = []
        self.pos_frac_ = np.array([])
        self.spheric_labels_ = []
        
        for fname in basis_fname:
            assert os.path.exists(fname), f'{fname} does not exist'
            self.basis_.append(lcaoReader(fname))
        self.atom_parser()

    def atom_parser(self):
        atom_str = str(self.cell_.atom)
        atom_list = atom_str.split(';')
        species = []
        pos = []
        for atom in atom_list:
            atom = atom.strip()
            atom = atom.split()
            assert len(atom) == 4, f'atom {atom} is not in the correct format'
            species.append(atom[0])
            pos.append([float(x) for x in atom[1:]])
        pos = np.stack(pos) # (natom, 3)
        pos_frac = pos @ np.linalg.inv(np.asarray(self.cell_.a)) # (natom, 3)
        pos_frac = pos_frac % 1

        self.species_ = species   # (natom)
        self.pos_frac_ = pos_frac # (natom, 3)

        basis_element_type = [basis.element_type for basis in self.basis_]
        for atom in set(self.species_):
            assert atom in basis_element_type, f'{atom} is not in the basis set'

    def eval_ao(self,
                ):
        '''
        Evaluate the atomic orbitals on the FFT grid

        @return
        np.ndarray: (nbasis, nx, ny, nz)
        '''
        with torch.no_grad():
            fftw = torch.tensor(self.fft_)
            grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, steps=self.fft_[0]),
                                              torch.linspace(0, 1, steps=self.fft_[1]),
                                              torch.linspace(0, 1, steps=self.fft_[2]), indexing='ij'), -1)
            grid = grid.reshape(-1, 3)  # [ngrid, 3]
            ngrid = grid.shape[0]
            pos_frac = torch.tensor(self.pos_frac_, dtype=torch.float32)
            cell_a = torch.tensor(self.cell_.a, dtype=torch.float32)

            diff = grid[None, :, :] - pos_frac[:, None, :]  # [natom, ngrid, 3]
            diff = (diff + 0.5) % 1 - 0.5
            diff = torch.matmul(diff, cell_a)  # [natom, ngrid, 3]
            r_norm = torch.norm(diff, dim=-1)  # [natom, ngrid]
            mask = r_norm < 11.0

            basis_o = {}
            for basis in self.basis_:
                species = basis.element_type
                idx = torch.from_numpy(np.asarray(self.species_) == species)  # Keeping NumPy for boolean indexing compatibility
                natom_species = int(torch.sum(idx).item())
                r_norm_l = r_norm[idx][mask[idx]]
                r_norm_idx = torch.round(r_norm_l / basis.dr).long()  # [nvalid_grid * natom_species]
                r_norm_idx = torch.clamp(r_norm_idx, 0, basis.mesh - 1)

                results = []
                nlm = []
                l_pre = 0
                angular_part = None
                for ibasis, (key, value) in enumerate(basis.basis_set.items()):
                    (n, l) = key
                    if ibasis == 0 or l != l_pre:
                        diff_l = diff[idx][mask[idx]].roll(shifts=-1, dims=-1)
                        angular_part = spherical_harmonics(l, diff_l, normalize=True)
                        l_pre = l
                    basis_val_l = torch.tensor(value)[r_norm_idx][:, None] * angular_part
                    basis_val = torch.zeros(natom_species, ngrid, int(2 * l + 1), dtype=basis_val_l.dtype)
                    basis_val[mask[idx]] = basis_val_l
                    norm = torch.mean(basis_val ** 2, dim=1, keepdim=True) * self.cell_.vol
                    basis_val /= torch.sqrt(norm)
                    results.append(basis_val.reshape(natom_species, *fftw.tolist(), 2 * l + 1).permute(4, 0, 1, 2, 3))
                    for m in range(-l, l + 1):
                        nlm.append([n, l, m + l])
                basis_o[basis.element_type] = (nlm, torch.cat(results, dim=0))

            atom_mem = {}
            for atom in set(self.species_):
                atom_mem[atom] = 0

            name_f, basis_f = [], []
            for iatom, atom in enumerate(self.species_):
                nlm_l, results = basis_o[atom]
                for n, l, m in nlm_l:  # Convert back to NumPy if needed for indexing
                    l_str = list(param.ANGULAR)[l]
                    name_f.append(' '.join([str(iatom), atom]) + ' ' + ''.join([str(n + 1 + param.ANGULARMAP[l_str]), l_str, param.REAL_SPHERIC[l][m]]))
                basis_f.append(results[:, atom_mem[atom], :, :])
                atom_mem[atom] += 1

            self.spheric_labels_ = name_f
            return torch.cat(basis_f, dim=0).numpy()

    @property
    def spheric_labels(self):
        return self.spheric_labels_
