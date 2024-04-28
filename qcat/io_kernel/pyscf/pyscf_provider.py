import numpy as np
import os

from qcat.io_kernel.base.base_provider import BaseProvider
from pyscf.pbc.gto import Cell

class PYSCFProvider(BaseProvider):
    def __init__(self,
                 cell: Cell,
                 ):
        super().__init__()
        self.cell_ = cell

    @property
    def cell(self)->np.ndarray:
        return self.cell_.a

    @property
    def atom(self)->str:
        return self.cell_.atom
