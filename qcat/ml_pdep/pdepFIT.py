'''
Note: \chi_0 = pdepg.T @ np.diag(pdepeig) @ pdepg.conj(), where pdepg is function with shape [npdep, nmill]
'''
import json
import os
import numpy as np
from scipy.linalg import eigh
import math
from typing import List
from loguru import logger

from westpy import qe_io

from qcat.io_kernel import pyscfHelper, QEProvider
from qcat.utils import setLogger
from .core import *

setLogger()

class PDEP2AO(object):
    def __init__(self,
                 wfc_fname,
                 wstat_folder,
                 basis: str = "aug-cc-pvqz",
                 unit: str = "B",
                 exp_to_discard = None,
                 ):
        assert os.path.exists(wfc_fname), f"{wfc_fname} does not exist"
        assert os.path.exists(wstat_folder), f"{wstat_folder} does not exist"

        self.qe = qe_io(wfc_fname, wstat_folder)
        dft_provider = QEProvider(wfc_fname)
        self.pyscf_obj = pyscfHelper(dft_provider, basis, unit, exp_to_discard)
        self.gd4pi = np.array([])

    def getChiSpecDecomp(self):
        b = np.array([self.qe.b1, self.qe.b2, self.qe.b3])
        g_vec = self.qe.mill @ b      # [nmill, 3]
        g_vec_norm = np.linalg.norm(g_vec, axis=-1)
        eigval = self.qe.pdepeig / (1 - self.qe.pdepeig)
        self.gd4pi = g_vec_norm[None, :] / math.sqrt(4 * math.pi) # [1, nmill]
        return eigval, self.qe.pdepg * self.gd4pi # [npdep, nmill]

    @staticmethod
    def decom2Eigen(spec_val: np.ndarray, # [npdep]
                    spec_vec: np.ndarray, # [npdep, nmill],
                    ):
        chi_eigval, chi_eigvec = oeigh(spec_vec.T, spec_val) # [nmill, npdep]
        return chi_eigval, chi_eigvec.T # [npdep, nmill]


    def getAO_G(self,
                shls_slice=None,
                cutoff=None,
                use_lcao: bool=False,
                lcao_fname=None,
                remove_shls: List=['s', 'g', 'h', 'i', 'j'],
                ):
        basis_cpu = self.pyscf_obj.get_basis(shls_slice, cutoff, use_lcao, lcao_fname) # (nAO, nx, ny, nz)
        basis_cpu, labels = clear_basis(basis_cpu, self.pyscf_obj.spheric_labels, remove_shls)
        nbasis = basis_cpu.shape[0]
        logger.info(f"nbasis: {nbasis}")
        basis_cpu = np.fft.fftn(basis_cpu, axes=(1, 2, 3), norm='forward')
        basis_g = np.zeros((nbasis, len(self.qe.mill)), dtype=basis_cpu.dtype) # [nbasis, nmill]
        mill_x, mill_y, mill_z = self.qe.mill.T
        for i in range(nbasis):
            basis_g[i] = basis_cpu[i][mill_x, mill_y, mill_z]
        norm_basis_g = np.sqrt(np.diag((basis_g @ basis_g.T.conj()).real) * 2)
        basis_g /= norm_basis_g[:, None] # [nbasis, nmill]
        return basis_g, labels

    @staticmethod
    def compute_S(basis_g: np.ndarray, # [nbasis, nmill]
                  ):
        return (basis_g.conj() @ basis_g.T).real * 2

    @staticmethod
    def compute_QAQ(basis_g: np.ndarray, # [nbasis, nmill]
                    eigvec: np.ndarray,  # [npdep, ngrid]
                    eigval: np.ndarray,  # [npdep]
                    ):
        return ((basis_g.conj() @ eigvec.T).real * 2) @ np.diag(eigval) @ ((eigvec.conj() @ basis_g.T).real * 2)


    def run(self,
            workdir='./log',
            prefix='westpy',
            **kwargs):
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        chi_decom_eigval, chi_decom_eigvec = self.getChiSpecDecomp() # [npdep], [npdep, nmill]
        npdep = chi_decom_eigval.size
        chi_eigval, chi_eigvec = self.decom2Eigen(chi_decom_eigval, chi_decom_eigvec) # [npdep], [npdep, nmill]
        basis_g, labels = self.getAO_G(**kwargs) # [nbasis, nmill]
        S = self.compute_S(basis_g) # [nbasis, nbasis]
        QAQ = self.compute_QAQ(basis_g, chi_eigvec, chi_eigval) # [nbasis, nbasis]

        label_fname = os.path.join(workdir, 'orbital_labels.json')
        with open(label_fname, 'w') as f:
            json.dump(labels.tolist(), f, indent=4)
        logger.info(f"Orbital labels are saved in {label_fname}")

        s_fname = os.path.join(workdir, 'S.npy')
        np.save(s_fname, S)
        logger.info(f"S matrix is saved in {s_fname}")

        qaq_fname = os.path.join(workdir, 'QAQ.npy')
        np.save(qaq_fname, QAQ)
        logger.info(f"QAQ matrix is saved in {qaq_fname}")

        chi_eigval_fit, coeff = eigh(QAQ, S)
        chi_eigvec_fit = coeff.T @ basis_g # [nbasis, nmill]
        chitil_decom_eigval_fit = chi_eigval_fit
        chitil_decom_eigvec_fit = np.divide(chi_eigvec_fit,
                                            self.gd4pi,
                                            out=np.zeros_like(chi_eigvec_fit),
                                            where=self.gd4pi!=0) # [nbasis, nmill]
        chitil_eigval_fit, chitil_eigvec_fit = self.decom2Eigen(chitil_decom_eigval_fit, chitil_decom_eigvec_fit) # [nbasis], [nbasis, nmill]
        chi0til_eigval_fit = chitil_eigval_fit / (1 + chitil_eigval_fit)

        chi0til_eigval_fit = chi0til_eigval_fit[:npdep]
        chitil_eigvec_fit = chitil_eigvec_fit[:npdep]

        self.qe.write_wstat(chi0til_eigval_fit, chitil_eigvec_fit, prefix=prefix, eig_mat='chi_0')
        return chi0til_eigval_fit, chitil_eigvec_fit
