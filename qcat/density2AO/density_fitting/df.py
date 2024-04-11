from pyscf import pbc
from pyscf.pbc.dft import numint
from typing import List, Any
import numpy as np

from qcat.density2AO import CubeReader
from qcat.density2AO import setup_logger
from qcat.basis import lcaoGenerator
from loguru import logger

class DF:
    def __init__(self,
                 filename: str,
                 basis: str = "cc-pvdz-jkfit",
                 unit: str = "B",
                 roll: bool = True,
                 exp_to_discard = None,
                 debug: bool = False,
                 ):
        self.filename_ = filename
        self.density_ = CubeReader(filename, roll)
        self.pyscf_obj_ = pbc.gto.Cell()
        self.coeff_ = np.empty(0)
        self.f_density_ = np.empty(0)
        self.build_cell(basis=basis,
                        unit=unit,
                        exp_to_discard=exp_to_discard)
        if debug:
            setup_logger(level='DEBUG')
        else:
            setup_logger(level='INFO')

    def build_cell(self,
                   basis: str = "cc-pvdz-jkfit",
                   unit: str = "B",
                   exp_to_discard = None,
                   ):
        cell = self.density_.cell
        atom = self.density_.atom
        self.pyscf_obj_.atom = atom
        self.pyscf_obj_.a = cell
        self.pyscf_obj_.unit = unit
        self.pyscf_obj_.basis = basis
        self.pyscf_obj_.exp_to_discard = exp_to_discard
        self.pyscf_obj_.build()

    def get_basis(self,
                  shls_slice=None,
                  cutoff=None,
                  use_lcao: bool = False,
                  lcao_fname = None,
                  )->np.ndarray:
        nxyz = self.density_.data.shape
        if not use_lcao:
            grid = self.pyscf_obj_.get_uniform_grids(nxyz)

            # Compute AO values on the grid
            basis_cpu = np.asarray(numint.eval_ao(self.pyscf_obj_, grid, shls_slice=shls_slice, cutoff=cutoff))
            basis_cpu = basis_cpu.reshape([*list(nxyz), -1])
            basis_cpu = np.transpose(basis_cpu, axes=(3, 0, 1, 2)) # (nAO, nx, ny, nz)
        else:
            assert lcao_fname is not None, "LCAO basis file is not provided"
            lcao = lcaoGenerator(cell=self.cell, basis_fname=lcao_fname, fftw=nxyz)
            basis_cpu = lcao.eval_ao()
        return basis_cpu

    @staticmethod
    def compute_overlap(pyscf_obj: pbc.gto.Cell,
                        analytical: bool = True,
                        basis: np.ndarray = None,
                        )->np.ndarray:
        if analytical:
            ovm = np.asarray(pyscf_obj.pbc_intor('int1e_ovlp_sph'))
        else:
            if basis is None:
                logger.error("Basis set is not provided")
                raise ValueError
            volume = np.linalg.det(np.asarray(pyscf_obj.a))
            ovm = np.tensordot(basis, basis, axes=([1, 2, 3], [1, 2, 3])) * volume / np.asarray(basis.shape[1:]).prod()
        return ovm

    @staticmethod
    def compute_w(basis: np.ndarray,
                  vol_data: np.ndarray,
                  volume: float,
                  )->np.ndarray:
        w = np.einsum('ijkl, jkl -> i', basis, vol_data) * volume / np.asarray(basis.shape[1:]).prod()
        w = w[:, np.newaxis]
        return w

    @staticmethod
    def fitted_density(coeff: np.ndarray,
                       basis: np.ndarray,
                       )->np.ndarray:
        nxyz = basis.shape[1:]
        df = coeff.T @ basis.reshape((basis.shape[0], -1))
        df = df.reshape(nxyz)
        return df

    def compute_coeff(self,
                      analytical: bool = True,
                      basis = None,
                      shls_slice=None,
                      cutoff=None,
                      use_lcao: bool = False,
                      lcao_fname = None,
                      )->np.ndarray:
        # Compute the basis set
        if basis is None:
            basis = self.get_basis(shls_slice=shls_slice,
                                   cutoff=cutoff,
                                   use_lcao=use_lcao,
                                   lcao_fname=lcao_fname) # basis on cpu

        # Compute the overlap matrix
        ovm = self.compute_overlap(self.cell, analytical, basis)

        # Compute the W matrix
        w = self.compute_w(basis, self.density_.data, self.volume)

        # Compute the coefficients
        self.coeff_ = np.linalg.inv(ovm) @ w

        # Compute the fitted density
        self.f_density_ = self.fitted_density(self.coeff_, basis)

        return self.coeff_

    @property
    def spheric_labels(self,
                       )->List:
        return self.pyscf_obj_.spheric_labels()

    @property
    def cell(self,
             )->pbc.gto.Cell:
        return self.pyscf_obj_

    @property
    def coeff(self,
              )->np.ndarray:
        return self.coeff_

    @property
    def o_density(self,
                 )->np.ndarray:
        return self.density_.data

    @property
    def f_density(self,
                 )->np.ndarray:
        return self.f_density_

    @property
    def volume(self,
            )->Any:
        return np.linalg.det(np.asarray(self.cell.a))
