import numpy as np

class BaseProvider:
    def __init__(self,
                 filename=None,
                 ):
        self.filename_ = filename
        self.cell_ = np.empty(0)
        self.atom_ = ""
        self.nxyz_ = np.array([0, 0, 0])

    def parse_file(self):
        pass

    def __str__(self):
        return f"Filename:\n{self.filename_}\n\nCell:\n{self.cell_}\n\nAtom:\n{self.atom_}\n\nNxyz:\n{self.nxyz_}"

    @property
    def cell(self)->np.ndarray:
        return self.cell_

    @property
    def atom(self)->str:
        return self.atom_

    @property
    def nxyz(self)->np.ndarray:
        return self.nxyz_

