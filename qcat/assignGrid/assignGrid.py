import numpy as np
from .kernel import generate_mnl
from typing import Tuple
import time
from qcat.utils import setLogger
from loguru import logger

setLogger()

class assignGrid(object):
    def __init__(self,
                 cell: np.ndarray,
                 fftw: Tuple[int, int, int],
                 atom_pos: np.ndarray,
                 rcut: float,
                 ):
        self.cell_ = cell
        self.fftw_ = fftw
        self.atom_pos_ = atom_pos
        self.rcut_ = rcut

        self.a_ = np.diag(np.array([1.0 / f for f in self.fftw_])) @ self.cell_
        self.adjust_rcut_ = self.rcut_ + np.sum(np.linalg.norm(self.a_, axis=-1))
        b1 = self.cell_[0]
        b1 = b1 / np.linalg.norm(b1)

        b3 = np.cross(self.cell_[0], self.cell_[1])
        b3 = b3 / np.linalg.norm(b3)

        b2 = np.cross(b3, b1)
        b2 = b2 / np.linalg.norm(b2)
        self.b_ = np.array([b1, b2, b3])

    def fold_within_a(self):
        rel_atom_pos = np.linalg.inv(self.cell_) @ self.atom_pos_
        rel_atom_pos = np.mod(rel_atom_pos, 1)
        rel_atom_pos = np.array([rel_atom_pos[i] * self.fftw_[i] for i in range(3)])
        shift = np.floor(rel_atom_pos).astype(int)
        rel_atom_pos = np.mod(rel_atom_pos, 1)
        return rel_atom_pos, shift

    def compute_idx(self):
        '''
        if:
        a1 = alpha11 * b1
        a2 = alpha21 * b1 + alpha22 * b2
        a3 = alpha31 * b1 + alpha32 * b2 + alpha33 * b3
        then:
        m * a1 + n * a2 + l * a3 = (m * alpha11 + n * alpha21 + l * alpha31) * b1 + (n * alpha22 + l * alpha32) * b2 + l * alpha33 * b3
        '''
        start = time.time()
        _, shift = self.fold_within_a()
        atom_pos_frac = self.atom_pos_[None, :] @ np.linalg.inv(self.cell_)
        alpha = (self.a_ @ np.linalg.inv(self.b_)).astype(np.float64)

        logger.info(f"start generate_mnl: {time.time() - start:.3f}s")

        mnl = generate_mnl(alpha, self.adjust_rcut_)

        logger.info(f"end generate_mnl: {time.time() - start:.3f}s")

        mnl = np.asarray(mnl).astype(int)
        mnl = mnl + shift[None, :]
        mnl = np.mod(mnl, self.fftw_)
        mnl_fractional = mnl / np.array(self.fftw_)[None, :]
        diff_frac = (atom_pos_frac - mnl_fractional + 0.5) % 1 - 0.5
        diff = np.linalg.norm(diff_frac @ self.cell_, axis=-1)
        mask = diff <= self.rcut_
        mnl = mnl[mask] # [ngrid, 3]
        logger.info(f"end: {time.time() - start:.3f}s")
        return mnl

    def __str__(self):
        return f" cell:\n{self.cell_},\n\n fftw:\n{self.fftw_},\n\n atom_pos:\n{self.atom_pos_},\n\n rcut:\n{self.rcut_}"
