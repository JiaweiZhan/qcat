from pyscf import pbc
from typing import List, Tuple
import os
import numpy as np
from scipy.special import sph_harm

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

    @staticmethod
    def real_sph_harm(m, l, phi, theta):
        '''
        n | l | m quantum number
        phi: azimuthal angle
        theta: polar angle
        '''
        if m == 0:
            angular_part = sph_harm(m, l, phi, theta).real
        elif m < 0:
            angular_part = complex(0, 1) * np.sqrt(0.5) * (sph_harm(m, l, phi, theta) - (-1) ** (-m) * sph_harm(-m, l, phi, theta))
            angular_part = angular_part.real
        else:
            angular_part = np.sqrt(0.5) * (sph_harm(-m, l, phi, theta) + (-1) ** m * sph_harm(m, l, phi, theta))
            angular_part = angular_part.real
        return angular_part

    def eval_ao(self,
                ):
        fftw = self.fft_
        grid = np.mgrid[0:fftw[0], 0:fftw[1], 0:fftw[2]]
        grid = grid.astype(np.float64)
        grid[0] /= fftw[0]
        grid[1] /= fftw[1]
        grid[2] /= fftw[2]

        grid = np.transpose(grid, (1, 2, 3, 0))  # [x, y, z, 3]
        grid = grid.reshape(-1, 3) # [ngrid, 3]
        diff = grid[None, :, :] - self.pos_frac_[:, None, :] # [natom, ngrid, 3]
        diff = (diff + 0.5) % 1 - 0.5
        diff = diff @ np.asarray(self.cell_.a)              # [natom, ngrid, 3]
        r_norm = np.linalg.norm(diff, axis=-1)              # [natom, ngrid]
        theta = np.arccos(np.divide(diff[:, :, 2], r_norm, out=np.zeros_like(r_norm), where=r_norm!=0)) # [natom, ngrid]
        phi = np.arctan2(diff[:, :, 1], diff[:, :, 0])                                                  # [natom, ngrid]

        basis_o = {}
        for basis in self.basis_:
            species = basis.element_type
            idx = np.asarray(self.species_) == species
            natom_species = np.sum(idx)
            r_norm_idx = np.round(r_norm[idx] / basis.dr).astype(np.int32) # [natom_species, ngrid]
            r_norm_idx = np.clip(r_norm_idx, 0, basis.mesh - 1)

            phi_l = phi[idx]
            theta_l = theta[idx]

            results = []
            atom_idx = []
            nlm = []
            for key, value in basis.basis_set.items():
                (n, l) = key
                for m in range(-l, l + 1):
                    angular_part = self.real_sph_harm(m, l, phi_l, theta_l)
                    basis_val = value[r_norm_idx] * angular_part
                    norm = np.mean(np.square(basis_val)) * self.cell_.vol
                    basis_val /= np.sqrt(norm)
                    results.append(basis_val.reshape(natom_species, *fftw))
                    for i in range(natom_species):
                        atom_idx.append(i + 1)
                        nlm.append([str(n), str(l), str(m)])
            basis_o[basis.element_type] = (np.asarray(atom_idx), np.asarray(nlm), np.concatenate(results))

        atom_mem = {}
        for atom in set(self.species_):
            atom_mem[atom] = 1

        name_f, basis_f = [], []
        for atom in self.species_:
            basis_a_idx = basis_o[atom][0] == atom_mem[atom]
            nlm_l = basis_o[atom][1][basis_a_idx]
            name = [' '.join([str(atom_mem[atom]), atom, n, l, m]) for n, l, m in nlm_l]
            name_f.extend(name)
            basis_f.append(basis_o[atom][2][basis_a_idx])
            atom_mem[atom] += 1
        self.spheric_labels_ = name_f
        return np.concatenate(basis_f)

    @property
    def spheric_labels(self):
        return self.spheric_labels_
