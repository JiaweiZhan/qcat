import numpy as np
from typing import List
import re
from loguru import logger

from qcat.utils import setLogger

setLogger()


def find_characters_regex(input_string:str,
                          char_list: List=['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j'],
                          ):
    pattern = f"[{''.join(char_list)}]"
    if re.search(pattern, input_string):
        return True
    else:
        return False

def clear_basis(basis: np.ndarray, # (nbasis, nx, ny, nz)
                labels: List[str], # (nbasis,)
                shls: List=['s', 'g', 'h', 'i', 'j'],
                ):
    orbitals = np.array([not find_characters_regex(a.split()[-1].strip(), shls) for a in labels])
    return basis[orbitals], np.asarray(labels)[orbitals]

def oeigh(spectrum_phi: np.ndarray, # (ngrid, npdep)
          spectrum_eig: np.ndarray, # (npdep,)
          max_iter: int=100,
          tol: float=1e-10,
          k=None,
          first_zero:bool = True):
    '''
    Given a matrix A = phi * diag(eig) * phi.T.conj(), this function computes the first k eigenvalues and eigenvectors of A
    '''
    if first_zero:
        spectrum_phi = spectrum_phi[1:]
    ngrid, npdep = spectrum_phi.shape
    if k is None:
        k = npdep
    Q = np.random.rand(ngrid, k) + 1j * np.random.rand(ngrid, k) # Initial guess with complex numbers
    Q, _ = np.linalg.qr(Q)  # Orthonormalize the initial guess

    eigs = None
    for iteration in range(max_iter):
        A_Q = spectrum_phi @ (((spectrum_phi.conj().T @ Q).real * 2.0) * spectrum_eig[:, None])  # Action of A on Q

        ngrid = A_Q.shape[0]
        if first_zero:
            A_Q = np.vstack((A_Q, A_Q.conj()))
        else:
            A_Q = np.vstack((A_Q, A_Q[1:].conj()))
        Q, _ = np.linalg.qr(A_Q)
        Q = Q[:ngrid]
        A_Q = A_Q[:ngrid]

        projections_Q = (spectrum_phi.conj().T @ Q).real * 2
        scaled_projections_Q = projections_Q * spectrum_eig[:, None]
        T = (Q.conj().T @ (spectrum_phi @ scaled_projections_Q)).real * 2
        eigs, vecs = np.linalg.eigh(T)
        Q = Q @ vecs

        # Use the estimated eigenvalues 'eigs' in the convergence criterion
        A_Q = spectrum_phi @ (((spectrum_phi.conj().T @ Q).real * 2) * spectrum_eig[:, None])  # Action of A on Q
        diff = np.linalg.norm(A_Q - Q @ np.diag(eigs), ord='fro')

        logger.info(f"Iteration {iteration+1}, Diff: {diff}")

        if diff < tol:
            logger.info("Convergence achieved.")
            break
        elif iteration == max_iter - 1:
            logger.info("Reached maximum iterations without convergence.")

    if first_zero:
        Q = np.pad(Q, ((1, 0), (0, 0)), mode='constant', constant_values=0.0)
    return eigs, Q
