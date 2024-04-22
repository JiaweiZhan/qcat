from pyscf import pbc
from typing import List
import numpy as np

from qcat.io_kernel import CubeProvider
from qcat.utils import setLogger
from loguru import logger
from qcat.io_kernel import pyscfHelper

setLogger(level='INFO')

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
        self.density_ = CubeProvider(filename, roll)
        self.pyscf_wrapper = pyscfHelper(self.density_, basis=basis,
                                         unit=unit, exp_to_discard=exp_to_discard)
        self.coeff_ = np.empty(0)
        self.f_density_ = np.empty(0)
        if debug:
            setLogger(level='DEBUG')
        else:
            setLogger(level='INFO')

    def get_basis(self,
                  shls_slice=None,
                  cutoff=None,
                  use_lcao: bool = False,
                  lcao_fname = None,
                  )->np.ndarray:
        return self.pyscf_wrapper.get_basis(shls_slice=shls_slice, cutoff=cutoff,
                                            use_lcao=use_lcao, lcao_fname=lcao_fname)

    @staticmethod
    def compute_overlap(pyscf_obj: pbc.gto.Cell,
                        analytical: bool = True,
                        basis=None,
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
        ovm = self.compute_overlap(self.pyscf_wrapper.cell, analytical, basis)

        # Compute the W matrix
        w = self.compute_w(basis, self.density_.data, self.pyscf_wrapper.cell.vol)

        # Compute the coefficients
        self.coeff_ = np.linalg.inv(ovm) @ w

        # Compute the fitted density
        self.f_density_ = self.fitted_density(self.coeff_, basis)

        return self.coeff_

    @property
    def spheric_labels(self,
                       )->List:
        return self.pyscf_wrapper.spheric_labels

    @property
    def cell(self,
             )->pbc.gto.Cell:
        return self.pyscf_wrapper.cell

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
