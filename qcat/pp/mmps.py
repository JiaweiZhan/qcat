import os
from loguru import logger
import numpy as np
import pandas as pd

from qcat.io_kernel import QBOXRead
from qcat.utils import setLogger
from qcat.assignGrid import assignGrid

setLogger(filter_out="qcat.assignGrid.assignGrid")
threshold = 1.2

def default_rcut(atom_pos: np.ndarray, # atom pos in cartesian coordinate
                 cell: np.ndarray # cell vector
                ):
    natom = atom_pos.shape[0]
    atom_pos_frac = atom_pos @ np.linalg.inv(cell)
    atom_pos_frac %= 1
    dist_frac = atom_pos_frac[None, :, :] - atom_pos_frac[:, None, :]
    dist_frac = (dist_frac + 0.5) % 1 - 0.5
    dist = dist_frac.reshape((-1, 3)) @ cell
    dist = dist.reshape(natom, natom, 3)
    dist = np.linalg.norm(dist, axis=-1)
    dist[np.arange(natom), np.arange(natom)] = np.max(dist)
    min_dist = np.min(dist)
    rcut = min_dist / 2
    return rcut


def mag_moment_per_site(qbox_folder: str,
                        rcut = None,
                        ):
    assert os.path.exists(qbox_folder), f"{qbox_folder} does not exist."
    qbox_reader = QBOXRead(qbox_folder)
    qbox_reader.parse_info()
    info_dict = qbox_reader.parse_wfc()

    nspin, fftw, nks, wfc_file, atompos, cell = info_dict["nspin"], info_dict["fftw"], info_dict["nks"], info_dict["wfc_file"], info_dict["atompos"], info_dict["cell"]
    logger.info(f"nspin: {nspin}, fftw: {fftw}, nks: {nks}")
    if nspin == 1:
        raise ValueError("This function only works for spin-polarized calculation.")
    atom_pos_cart = np.array([pos[1:] for pos in atompos])

    rcut = default_rcut(atom_pos_cart, cell) if rcut is None else rcut
    logger.info(f"rcut: {rcut:^6.2e}")
    rm = rcut / threshold

    frac_coords_x, frac_coords_y, frac_coords_z = np.meshgrid(*[np.arange(i) / i for i in fftw], indexing="ij")
    frac_coords = np.stack([frac_coords_x, frac_coords_y, frac_coords_z], axis=-1) # shape (fftw0, fftw1, fftw2, 3)
    srho = np.zeros([nspin] + fftw.tolist())
    for ispin in range(nspin):
        for ik in range(nks):
            for fname in wfc_file[ispin][ik]:
                srho[ispin] += np.square(np.load(fname))

    per_site_info = {"atom": [], "charge": [], "mag_mom": []}
    for idx, atom_pos in enumerate(atom_pos_cart):
        atom_name = atompos[idx][0]
        atoms_pos_frac = atom_pos[None, :] @ np.linalg.inv(cell)
        atom_pos_frac = atoms_pos_frac % 1
        ag = assignGrid(cell, fftw, atom_pos, rcut)
        idx = ag.compute_idx()
        l, m, n = idx.T
        near_grid = frac_coords[l, m, n] # [n_near_grid, 3]
        dist = (near_grid - atom_pos_frac + 0.5) % 1 - 0.5
        dist = np.linalg.norm(dist @ cell, axis=-1)
        weight = np.where(dist < rm, 1.0, 1 - (dist - rm) / (0.2 * rm))
        spinup = np.sum(srho[0][l, m, n] * weight) / np.prod(fftw)
        spindown = np.sum(srho[1][l, m, n] * weight) / np.prod(fftw)

        per_site_info["atom"].append(atom_name)
        per_site_info["charge"].append(spinup + spindown)
        per_site_info["mag_mom"].append(spinup - spindown)
    df = pd.DataFrame(per_site_info)
    qbox_reader.clean_wfc()
    return df

