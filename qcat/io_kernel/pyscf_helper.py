from pyscf import pbc
from pyscf.pbc.dft import numint
import numpy as np
from typing import List

from qcat.io_kernel.dft_provider.base_provider import BaseProvider
from qcat.basis import lcaoGenerator

class pyscfHelper:
    def __init__(self,
                 dft_provider: BaseProvider,
                 basis: str = "aug-cc-pvqz",
                 unit: str = "B",
                 exp_to_discard = None,
                 ):
        self.dft_provider_ = dft_provider
        self.pyscf_cell_ = pbc.gto.Cell()
        self.basis_wrapper = None
        self.use_pyscf_basis = True
        self.build_cell(basis, unit, exp_to_discard)

    def build_cell(self,
                   basis: str = "aug-cc-pvqz",
                   unit: str = "B",
                   exp_to_discard = None,
                   ):
        self.pyscf_cell_.atom = self.dft_provider_.atom
        self.pyscf_cell_.a = self.dft_provider_.cell
        self.pyscf_cell_.unit = unit
        self.pyscf_cell_.basis = basis
        self.pyscf_cell_.exp_to_discard = exp_to_discard
        self.pyscf_cell_.build()

    def get_basis(self,
                  shls_slice=None,
                  cutoff=None,
                  use_lcao: bool=False,
                  lcao_fname=None,
                  )->np.ndarray:
        nxyz = self.dft_provider_.nxyz
        if not use_lcao:
            self.use_pyscf_basis = True
            grid = self.pyscf_cell_.get_uniform_grids(nxyz)

            # Compute AO values on the grid
            basis_cpu = numint.eval_ao(self.pyscf_cell_, grid, shls_slice=shls_slice, cutoff=cutoff)
            basis_cpu = basis_cpu.reshape([*(nxyz.tolist()), -1])
            basis_cpu = basis_cpu.transpose(3, 0, 1, 2) # (nAO, nx, ny, nz)
        else:
            self.use_pyscf_basis = False
            assert lcao_fname is not None, "LCAO basis file is not provided"
            self.basis_wrapper = lcaoGenerator(cell=self.pyscf_cell_, basis_fname=lcao_fname, fftw=tuple(nxyz))
            basis_cpu = self.basis_wrapper.eval_ao()
        return basis_cpu

    @property
    def spheric_labels(self,
                       )->List:
        if self.use_pyscf_basis:
            return self.pyscf_cell_.spheric_labels()
        else:
            return self.basis_wrapper.spheric_labels

    @property
    def cell(self,
             )->pbc.gto.Cell:
        return self.pyscf_cell_
