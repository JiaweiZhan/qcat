from ase import data
import numpy as np
from .base_provider import BaseProvider

pt = data.chemical_symbols

class CubeProvider(BaseProvider):
    def __init__(self,
                 filename: str,
                 roll: bool = True):
        super().__init__(filename)
        self.filename_ = filename
        self.data_ = np.empty(0)
        self.parse_file(roll)

    def parse_file(self,
                   roll: bool = True):
        with open(self.filename_, 'r') as f:
            # Read the first 2 lines
            for _ in range(2):
                f.readline()
            # read natom
            line = f.readline()
            natom = int(line.split()[0])

            # read nx, ny, nz
            cell_info = np.array([[float(i) for i in f.readline().split()] for _ in range(3)])
            nxyz = cell_info[:, 0].astype(int)
            self.nxyz_ = nxyz
            cell = cell_info[:, 1:]
            self.cell_ = np.multiply(cell, nxyz[:, None].repeat(3, axis=1))

            # read atoms
            atom_ = []
            for _ in range(natom):
                line = f.readline()
                line = line.split()
                atom_symbol = pt[int(line[0])]
                pos = line[2:]
                atom_info = atom_symbol + " " + " ".join(pos)
                atom_.append(atom_info)
            self.atom_ = "; ".join(atom_)

            # read data
            self.data_ = np.fromfile(f, sep=" ", count=nxyz.prod())

        self.data_ = self.data_.reshape(nxyz)
        if roll:
            self.data_ = np.roll(self.data_, nxyz[0] // 2, axis=0)
            self.data_ = np.roll(self.data_, nxyz[1] // 2, axis=1)
            self.data_ = np.roll(self.data_, nxyz[2] // 2, axis=2)

    @property
    def data(self)->np.ndarray:
        return self.data_
