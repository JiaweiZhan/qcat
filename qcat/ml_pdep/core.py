import numpy as np
from typing import List
import re
from loguru import logger
import torch
import scipy

from qcat.utils import setLogger

setLogger()

def clear_basis(labels: List[str], # (nbasis,)
                shls: List=['s', 'g', 'h', 'i', 'j'],
                ):
    """
    Filters the basis and labels based on absence of specified shell types.
    """
    # Compile regex once
    char_pattern = re.compile(f"[{''.join(shls)}]")

    # Create a mask with vectorized operations
    mask = np.array([bool(char_pattern.search(label.split()[-1].strip())) for label in labels])
    mask = ~mask

    # Apply mask to basis and labels
    return np.array(labels)[mask], mask

def oeigh(spectrum_phi: np.ndarray, # (ngrid, npdep)
          spectrum_eig: np.ndarray, # (npdep,)
          max_iter: int=10,
          tol: float=1e-10,
          k=None,
          first_zero:bool = True):
    '''
    Given a matrix A = phi * diag(eig) * phi.T.conj(), this function computes the first k eigenvalues and eigenvectors of A
    '''
    with torch.no_grad():
        if first_zero:
            spectrum_phi = spectrum_phi[1:]
        ngrid, npdep = spectrum_phi.shape
        if k is None:
            k = npdep
        phi_dtype = spectrum_phi.dtype
        eig_dtype = spectrum_eig.dtype
        dtype_torch = getattr(torch, 'cdouble')
        if phi_dtype == np.complex128:
            dtype_torch = getattr(torch, 'cdouble')
        else:
            dtype_torch = getattr(torch, 'cfloat')
        spectrum_phi = torch.from_numpy(spectrum_phi)
        spectrum_eig = torch.from_numpy(spectrum_eig)
        mem_req = 3 * ngrid * npdep * spectrum_phi.element_size() / 1024**3
        logger.info(f"Memory required: {mem_req:.2f} GB")
        Q = torch.rand(ngrid, k, dtype=spectrum_phi.dtype)
        Q, _ = torch.linalg.qr(Q)  # Orthonormalize the initial guess

        eigs = None
        for iteration in range(max_iter):
            # <phi|Q>
            phiEigMat = (spectrum_phi.conj().T @ Q).real.to(dtype_torch) * 2
            A_Q = spectrum_phi @ torch.diag(spectrum_eig).to(dtype_torch) @ phiEigMat  # Action of A on Q

            ngrid = A_Q.shape[0]
            if first_zero:
                A_Q = torch.vstack((A_Q, A_Q.conj()))
            else:
                A_Q = torch.vstack((A_Q, A_Q[1:].conj()))
            Q, _ = torch.linalg.qr(A_Q)  # Orthonormalize the initial guess
            Q = Q[:ngrid]

            phiEigMat = (spectrum_phi.conj().T @ Q).real * 2
            T = phiEigMat.T @ torch.diag(spectrum_eig) @ phiEigMat
            eigs, vecs = torch.linalg.eigh(T)
            Q = Q @ vecs.to(dtype_torch)

            # Use the estimated eigenvalues 'eigs' in the convergence criterion
            phiEigMat = (spectrum_phi.conj().T @ Q).real.to(dtype_torch) * 2
            A_Q = spectrum_phi @ torch.diag(spectrum_eig).to(dtype_torch) @ phiEigMat  # Action of A on Q
            diff = np.linalg.norm(A_Q - Q @ torch.diag(eigs).to(dtype_torch), ord='fro')

            logger.info(f"Iteration {iteration+1}, Diff: {diff:^8.3e}")

            if diff < tol:
                logger.info("Convergence achieved.")
                break
            elif iteration == max_iter - 1:
                logger.info("Reached maximum iterations without convergence.")

        Q = Q.numpy().astype(phi_dtype)
        eigs = eigs.numpy().astype(eig_dtype)
    if first_zero:
        Q = np.pad(Q, ((1, 0), (0, 0)), mode='constant', constant_values=0.0)
    return eigs, Q

def reduce_noise_SVD(qaq: np.ndarray,
                     ):
    logger.info("Reduce noise by SVD...")
    with torch.no_grad():
        qaq = torch.from_numpy(qaq)
        U, S, Vh = torch.linalg.svd(qaq)
        nS = S.numel()
        mask = torch.ones_like(S, dtype=int)
        # use binary search to find the threshold, dependent on whether there is positive eigenvalues
        left, right = 0, nS - 1
        while left <= right:
            mid = (left + right) // 2
            mask_copy = mask.clone()
            mask_copy[mid:] = 0
            qaq_denoise = U @ torch.diag(S * mask_copy) @ Vh
            eigvals, _ = torch.linalg.eigh(qaq_denoise)
            if torch.any(eigvals > 1.e-4):
                right = mid - 1
            else:
                left = mid + 1
        if left == 0:
            logger.warning("No negative eigenvalues found.")
        mid = max(left, 1)
        mask_copy = mask.clone()
        mask_copy[mid:] = 0
        qaq_denoise = U @ torch.diag(S * mask_copy) @ Vh
        mask_copy[-1] = 1
        nonzeros = torch.sum(mask_copy).item()
        logger.info(f"nonzeros: {nonzeros} / {nS}")
    return qaq_denoise.numpy(), mask_copy.numpy().astype(bool)

