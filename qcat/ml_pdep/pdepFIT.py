"""
Note: \chi_0 = pdepg.T @ np.diag(pdepeig) @ pdepg.conj(), where pdepg is function with shape [npdep, nmill]
"""

import json
import os
import numpy as np
from scipy.linalg import eigh
import math
from typing import List
from loguru import logger
import torch
import time

from westpy import qe_io

from qcat.io_kernel import pyscfHelper, QEProvider
from qcat.utils import setLogger
from .core import *

setLogger()


class PDEP2AO(object):
    def __init__(
        self,
        wfc_fname,
        wstat_folder=None,
        basis: str = "cc-pvqz",
        unit: str = "B",
        exp_to_discard=0.1,
    ):
        assert os.path.exists(wfc_fname), f"{wfc_fname} does not exist"
        if wstat_folder is not None:
            assert os.path.exists(wstat_folder), f"{wstat_folder} does not exist"

        self.qe = qe_io(wfc_fname, wstat_folder)
        dft_provider = QEProvider(wfc_fname)
        self.pyscf_obj = pyscfHelper(dft_provider, basis, unit, exp_to_discard)
        b = np.array([self.qe.b1, self.qe.b2, self.qe.b3])
        g_vec = self.qe.mill @ b  # [nmill, 3]
        g_vec_norm = np.linalg.norm(g_vec, axis=-1)
        self.gd4pi = g_vec_norm[None, :] / math.sqrt(4 * math.pi)  # [1, nmill]

    def getChiSpecDecomp(self):
        eigval = self.qe.pdepeig / (1 - self.qe.pdepeig)
        return eigval, self.qe.pdepg * self.gd4pi  # [npdep, nmill]

    @staticmethod
    def decom2Eigen(
        spec_val: np.ndarray,  # [npdep]
        spec_vec: np.ndarray,  # [npdep, nmill],
        k=None,
        tol: float = 1e-10,
    ):
        chi_eigval, chi_eigvec = oeigh(
            spec_vec.T, spec_val, tol=tol, k=k
        )  # [nmill, npdep]
        return chi_eigval, chi_eigvec.T  # [npdep, nmill]

    def getAO_G(
        self,
        shls_slice=None,
        cutoff=None,
        use_lcao: bool = False,
        lcao_fname=None,
        remove_g: bool = True,
    ):
        if shls_slice is None:
            shls_slice = (0, self.pyscf_obj.cell.nbas)
        remove_shls = ["s", "h", "i", "j"]
        if remove_g:
            remove_shls.append("g")
        labels, mask = clear_basis(
            self.pyscf_obj.spheric_labels, remove_shls
        )
        logger.info(f"nbasis: {np.sum(mask)} / {len(mask)}")

        mill_x, mill_y, mill_z = self.qe.mill.T
        basis_g = []
        tot_prog = shls_slice[1] - shls_slice[0]
        prog_per = -1
        nstep = 5
        for i in range(shls_slice[0], shls_slice[1], nstep):
            shls_slice_l_upper = min(i + nstep, shls_slice[1])
            shls_slice_l = (i, shls_slice_l_upper)
            basis_cpu_l = self.pyscf_obj.get_basis(
                shls_slice_l, cutoff, use_lcao, lcao_fname
            )  # (nAO, nx, ny, nz)

            basis_cpu_l = torch.fft.fftn(
                torch.as_tensor(basis_cpu_l, dtype=torch.float64),
                dim=(1, 2, 3),
                norm="forward",
            )
            basis_g.append(basis_cpu_l[:, mill_x, mill_y, mill_z])
            cur_prog = shls_slice_l_upper - shls_slice[0]
            cur_prog_per = int(cur_prog / tot_prog * 5)
            if cur_prog_per > prog_per:
                prog_per = cur_prog_per
                logger.info(f"Progress: {prog_per*20:^3}%")
        basis_g = torch.vstack(basis_g)
        basis_g = basis_g[mask]
        norm_basis_g = torch.sqrt(torch.diag((basis_g @ basis_g.T.conj()).real) * 2)
        basis_g /= norm_basis_g[:, None]  # [nbasis, nmill]
        return basis_g, labels, mask

    def compute_S(
        self,
        basis_g: torch.Tensor,  # [nbasis, nmill]
        pyscf_overlap: bool = False,
        mask=None,
    ):
        if not pyscf_overlap:
            return (basis_g.conj() @ basis_g.T).real * 2
        else:
            ovm = self.pyscf_obj.cell.pbc_intor("int1e_ovlp_sph")
            return torch.as_tensor(ovm[mask, :][:, mask])

    @staticmethod
    def compute_QAQ(
        basis_g: torch.Tensor,  # [nbasis, nmill]
        eigvec : torch.Tensor,  # [npdep, nmill]
        eigval : torch.Tensor,  # [npdep]
    ):
        twoOrbitalMat = basis_g.conj() @ eigvec.T
        return (
            (twoOrbitalMat.real * 2)
            @ torch.diag(eigval)
            @ ((twoOrbitalMat.T.conj()).real * 2)
        )

    def compute_pdep(
        self,
        s: np.ndarray,
        qaq: np.ndarray,
        basis_g: np.ndarray,
        k=None,
        tol: float = 1e-1,
        npdep=None,
        noise_reduction: bool = False,
    ):
        mask_npdep = np.ones(qaq.shape[0]).astype(bool)
        if npdep is not None:
            mask_npdep[npdep:] = False
        if noise_reduction:
            qaq, mask_svd = reduce_noise_SVD(qaq)
            mask_npdep = np.logical_and(mask_npdep, mask_svd)
        chi_eigval_fit, coeff = eigh(qaq, s)
        chi_eigval_fit = chi_eigval_fit[mask_npdep]
        coeff = coeff[:, mask_npdep]
        if np.any(chi_eigval_fit > 0):
            logger.warning(
                f"Some eigenvalues of QAQ are positive. Reduce npdep from {npdep}."
            )
            logger.warning(f"Positive count:{np.sum(chi_eigval_fit > 0)}")
        chi_eigvec_fit = coeff.T @ basis_g  # [nbasis, nmill]
        chibar_decom_eigval_fit = chi_eigval_fit
        chibar_decom_eigvec_fit = np.divide(
            chi_eigvec_fit,
            self.gd4pi,
            out=np.zeros_like(chi_eigvec_fit),
            where=self.gd4pi != 0,
        )  # [nbasis, nmill]
        logger.info("Compute Eigen for chibar...")
        chibar_eigval_fit, chibar_eigvec_fit = self.decom2Eigen(
            chibar_decom_eigval_fit,
            chibar_decom_eigvec_fit,
            tol=tol,
            k=k,
        )  # [nbasis], [nbasis, nmill]
        chi0bar_eigval_fit = chibar_eigval_fit / (1 + chibar_eigval_fit)

        # chibar and chi0bar share same eigenvectors
        chi0bar_eigval_fit = chi0bar_eigval_fit
        chibar_eigvec_fit = chibar_eigvec_fit
        return chi0bar_eigval_fit, chibar_eigvec_fit

    @staticmethod
    def atomIdx(labels: np.ndarray):
        mat_dim = len(labels)
        atomIdx = []
        atom_start = 0
        for i in range(mat_dim):
            if i > 0 and int(labels[i].split()[0]) != int(labels[i - 1].split()[0]):
                # different atom start
                atomIdx.append((atom_start, i))
                atom_start = i
        # final atom
        atomIdx.append((atom_start, mat_dim))
        return atomIdx

    @staticmethod
    def one_center_DDRF(
        QAQ: np.ndarray,
        S: np.ndarray,
        atomIdx: List,
    ):
        eigval, coeff = eigh(QAQ, S)
        sigma_tilde = coeff @ np.diag(eigval) @ coeff.T
        qaq_1cddrf = np.zeros_like(QAQ)
        weight = np.zeros_like(sigma_tilde)
        for atom_start, atom_end in atomIdx:
            weight[:, :] = 0.0
            weight[atom_start:atom_end] = 0.5
            weight[:, atom_start:atom_end] = 0.5
            weight[atom_start:atom_end, atom_start:atom_end] = 1.0
            sigma_tilde_i = sigma_tilde * weight
            si_bar = S[atom_start:atom_end]
            si_inv = np.linalg.inv(S[atom_start:atom_end, atom_start:atom_end])
            lhs = si_bar.T @ si_inv @ si_bar
            qaq_1cddrf += lhs @ sigma_tilde_i @ lhs
        return qaq_1cddrf

    def run(
        self,
        outDir: str = "./log",
        pyscf_overlap: bool = False,
        qaq_threshold=None,
        method: str = "2c",
        precision: str = "double",
        tol: float = 1e-10,
        prefix: str = "westpy",
        compute_pdep: bool = True,
        **kwargs,
    ):
        start_time = time.time()

        possible_precision = ["float", "double"]
        assert (
            precision in possible_precision
        ), f"precision should be one of {possible_precision}"

        possible_method = ["2c", "1c"]
        assert method in possible_method, f"method should be one of {possible_method}"

        if not os.path.exists(outDir):
            os.makedirs(outDir)
        chi_decom_eigval, chi_decom_eigvec = (
            self.getChiSpecDecomp()
        )  # [npdep], [npdep, nmill]
        npdep = chi_decom_eigval.size
        basis_g, labels, mask = self.getAO_G(**kwargs)  # [nbasis, nmill]
        chi_decom_eigval = torch.as_tensor(chi_decom_eigval)
        chi_decom_eigvec = torch.as_tensor(chi_decom_eigvec)

        if precision == "float":
            basis_g = basis_g.to(torch.complex64)
            chi_decom_eigvec = chi_decom_eigvec.to(torch.complex64)
            chi_decom_eigval = chi_decom_eigval.to(torch.float32)

        S = self.compute_S(basis_g, pyscf_overlap, mask)  # [nbasis, nbasis]

        QAQ = self.compute_QAQ(
            basis_g, chi_decom_eigvec, chi_decom_eigval
        )  # [nbasis, nbasis]

        S = S.numpy()
        QAQ = QAQ.numpy()
        basis_g = basis_g.numpy()

        if method == "1c":
            atomIdx = self.atomIdx(labels)
            assert len(atomIdx) == self.pyscf_obj.cell.natm
            QAQ = self.one_center_DDRF(QAQ, S, atomIdx)

        if qaq_threshold:
            logger.info(f"Apply threshold {qaq_threshold:^5.2e} to QAQ matrix.")
            qaq_threshold = np.abs(qaq_threshold)
            QAQ = np.where(np.abs(QAQ) < qaq_threshold, 0.0, QAQ)
        sparse_ratio = (QAQ == 0.0).sum() / QAQ.size
        logger.info(f"QAQ matrix sparse ratio: {sparse_ratio*100:^5.2f}%.")

        label_fname = os.path.join(outDir, "orbital_labels.json")
        with open(label_fname, "w") as f:
            json.dump(labels.tolist(), f, indent=4)
        logger.info(f"Orbital labels are saved in {label_fname}")

        s_fname = os.path.join(outDir, "S.npy")
        np.save(s_fname, S)
        logger.info(f"S matrix is saved in {s_fname}")

        qaq_fname = os.path.join(outDir, "QAQ.npy")
        np.save(qaq_fname, QAQ)
        logger.info(f"QAQ matrix is saved in {qaq_fname}")

        if compute_pdep:
            pdep_eigval_fit, pdep_eigvec_fit = self.compute_pdep(
                s=S, qaq=QAQ, basis_g=basis_g, tol=tol, npdep=npdep
            )
            prefix = os.path.join(outDir, prefix)
            self.qe.write_wstat(
                pdep_eigval_fit, pdep_eigvec_fit, prefix=prefix, eig_mat="chi_0"
            )
        else:
            pdep_eigval_fit, pdep_eigvec_fit = None, None
        logger.info(f"Running Time: {time.time() - start_time:^8.2f}s")
        return pdep_eigval_fit, pdep_eigvec_fit


def tcddrf2PDEP(
    wfc_name: str,
    qaq: np.ndarray,
    s: np.ndarray,
    precision: str = "double",
    basis: str = "cc-pvqz",
    unit: str = "B",
    exp_to_discard=0.1,
    noise_reduction: bool = False,
    npdep=None,
    k=None,
    tol: float = 1e-10,
    outDir: str = "./log_tcddrf2PDEP",
    prefix: str = "tcddrf2PDEP",
    **kwargs,
):
    start_time = time.time()
    possible_precision = ["float", "double"]
    assert (
        precision in possible_precision
    ), f"precision should be one of {possible_precision}"
    pdep2ao = PDEP2AO(wfc_name, basis=basis, unit=unit, exp_to_discard=exp_to_discard)
    basis_g, _, _ = pdep2ao.getAO_G(**kwargs)
    basis_g = basis_g.numpy()

    if precision == "float":
        s = s.astype(np.float32)
        qaq = qaq.astype(np.float32)
        basis_g = basis_g.astype(np.complex64)
    pdep_eigval_fit, pdep_eigvec_fit = pdep2ao.compute_pdep(
        s=s,
        qaq=qaq,
        basis_g=basis_g,
        noise_reduction=noise_reduction,
        tol=tol,
        npdep=npdep,
        k=k,
    )
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    prefix = os.path.join(outDir, prefix)
    pdep2ao.qe.write_wstat(
        pdep_eigval_fit, pdep_eigvec_fit, prefix=prefix, eig_mat="chi_0"
    )
    logger.info(f"Running Time: {time.time() - start_time:^8.2f}s")
    return pdep_eigval_fit, pdep_eigvec_fit
