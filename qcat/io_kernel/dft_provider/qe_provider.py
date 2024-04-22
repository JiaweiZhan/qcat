import numpy as np
import os

from .base_provider import BaseProvider
from qcat.io_kernel import qe_io

class QEProvider(BaseProvider):
    def __init__(self,
                 filename: str,
                 ):
        super().__init__(filename)
        self.filename_ = filename
        self.parse_file()

    def parse_file(self):
        qe_outFolder = os.path.dirname(self.filename_)
        qe = qe_io.QERead(qe_outFolder)
        qe_dict = qe.parse_info(store=False)
        self.cell_ = qe_dict['cell']
        qe_atom = qe_dict['atompos']
        atom_str = []
        for comb in qe_atom:
            atom_name = comb[0]
            pos = comb[1: 4]
            pos_str = ' '.join([str(x) for x in pos])
            atom_str.append(f'{atom_name} {pos_str}')
        self.atom_ = "; ".join(atom_str)
        self.nxyz_ = np.asarray(qe_dict['npv'])
