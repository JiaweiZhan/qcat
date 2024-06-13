import numpy as np
import scipy as sp
from typing import List

def one_center_DDRF(
    QAQ: np.ndarray,
    S: np.ndarray,
    atomIdx: List,
):
    '''
    Zauchner, M.G., Horsfield, A. & Lischner, J.
    Accelerating GW calculations through machine-learned dielectric matrices. npj Comput Mater 9, 184 (2023).
    https://doi.org/10.1038/s41524-023-01136-y
    '''
    eigval, coeff = sp.linalg.eigh(QAQ, S)
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

def one_center_DDRF_v2(
    QAQ: np.ndarray,
    S: np.ndarray,
    atomIdx: List,
):
    '''
    Assume the density response caused by an AO can be fitted by AOs from
    the same atom. This mean:
    chi Psi_{A} = Psi C_{A}
    -> Psi^T chi Psi_{A} = Psi^T @ Psi_{A} @ S_{A}^{-1} @ ( Psi_{A}^T @ chi @ Psi_{A} )
    -> Psi_{A}^T chi Psi = ( Psi_{A}^T @ chi @ Psi_{A} ) @ S_{A}^{-1} @ Psi_{A}^T @ Psi
    '''
    # TODO: This function can be extened to include surroding atoms.

    qaq_1cddrf = np.zeros_like(QAQ)
    for atom_start, atom_end in atomIdx:
        qaq_1c = QAQ[atom_start:atom_end, atom_start:atom_end]
        s_1c_inv = np.linalg.inv(S[atom_start:atom_end, atom_start:atom_end])
        qaq_1cddrf[atom_start:atom_end] = qaq_1c @ s_1c_inv @ S[atom_start:atom_end]

    return 0.5 * (qaq_1cddrf + qaq_1cddrf.T)
