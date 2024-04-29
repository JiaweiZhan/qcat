import numpy as np
import re
import os
import pathlib
from scipy.sparse import csr_matrix
from typing import List, Dict
from ase.data import atomic_numbers, atomic_names
from ase.units import Bohr, Ang
import json, h5py

from qcat.utils import setLogger
from qcat.io_kernel.base.base_provider import BaseProvider
from loguru import logger

from qcat.io_kernel.deeph.deeph_utils import parse_matrix, restore_matrix
from qcat.io_kernel.deeph.get_rc import get_rc
from qcat.io_kernel.deeph.rotate import get_rh

setLogger()

str2l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

def write_R_sparse(mat: np.ndarray,
                   mat_option: str, # s: overlap, h: hamiltonian
                   outDir: str = './log',
                   ):
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    assert mat_option in ['s', 'h'], "mat_option should be 's' or 'h'"
    if mat_option == 's':
        fname = 'data-SR-sparse_SPIN0.csr'
    else:
        fname = 'data-HR-sparse_SPIN0.csr'
    fname = os.path.join(outDir, fname)
    csr_mat = csr_matrix(mat)
    logger.info(f"Writing {mat_option} matrix to {fname}")
    with open(fname, 'w') as file_obj:
        file_obj.write("STEP: 0\n")
        file_obj.write(f"Matrix Dimension of {mat_option.upper()}(R): {mat.shape[0]}\n")
        file_obj.write(f"Matrix Number of {mat_option.upper()}(R): 1\n")
        file_obj.write(f"0 0 0 {mat.size}\n")
        csr_mat.data.tofile(file_obj, sep=' ')
        file_obj.write("\n")
        csr_mat.indices.tofile(file_obj, sep=' ')
        file_obj.write("\n")
        csr_mat.indptr.tofile(file_obj, sep=' ')
    return

def label2orbital(labels: List,
                  save: bool = True,
                  outDir: str = './log',
                  ):
    if save:
        if not os.path.exists(outDir):
            os.makedirs(outDir)
    site_norbits_dict = {}
    orbital_types_dict = {}
    already_save = True
    norbital_pre = -1
    l_prev = -1
    element = []
    for i, label in enumerate(labels):
        idx_atom, atom_name, orbital = [a.strip() for a in label.split()]
        atom_num = atomic_numbers[atom_name]
        if i == 0 or (i > 0 and idx_atom != labels[i-1].split()[0].strip()):
            # different atom
            norbital_pre = -1
            l_prev = -1
            element.append(atom_num)
            if atom_num not in site_norbits_dict:
                already_save = False
            else:
                already_save = True
        if not already_save:
            site_norbits_dict[atom_num] = site_norbits_dict.get(atom_num, 0) + 1
            orbital_str = (re.search(r'[a-z]', orbital)).group(0)
            norbital = int((re.search(r'\d+', orbital)).group(0))
            if orbital_str not in str2l:
                logger.error(f"Unknown orbital type: {orbital}")
                raise NotImplementedError

            l = str2l[orbital_str]
            if norbital != norbital_pre or l != l_prev:
                orbital_types_dict[atom_num] = orbital_types_dict.get(atom_num, []) + [l]
                norbital_pre = norbital
                l_prev = l

    for k, v in site_norbits_dict.items():
        logger.info(f"Atom {atomic_names[k]} has {v} orbitals")

    if save:
        fname = os.path.join(outDir, 'orbital_types.dat')
        logger.info(f"Writing orbital types to {fname}")
        with open(fname, 'w') as f:
            for atomic_number in element:
                for index_l, l in enumerate(orbital_types_dict[atomic_number]):
                    if index_l == 0:
                        f.write(str(l))
                    else:
                        f.write(f"  {l}")
                f.write('\n')
        logger.info(f"Writing element to {os.path.join(outDir, 'element.dat')}")
        np.savetxt(os.path.join(outDir, "element.dat"), element, fmt='%d')
    return site_norbits_dict, orbital_types_dict, element

def write_sys_info(baseProvider: BaseProvider,
                   site_norbits_dict: Dict,
                   save: bool = True,
                   outDir: str = './log',
                   ):
    if save:
        if not os.path.exists(outDir):
            os.makedirs(outDir)
    bohr2agstrom = Bohr / Ang
    lattice = baseProvider.cell * bohr2agstrom
    atom_str = baseProvider.atom
    cart_coords = []
    site_norbits = []
    for atom in atom_str.split("; "):
        info = atom.strip().split()
        atom_num = atomic_numbers[info[0]]
        site_norbits.append(site_norbits_dict[atom_num])
        atom_pos = np.array([float(x) for x in info[1:]]) * bohr2agstrom
        cart_coords.append(atom_pos)
    cart_coords = np.vstack(cart_coords)

    if save:
        logger.info(f"Writing lattice to {os.path.join(outDir, 'lat.dat')}")
        np.savetxt(os.path.join(outDir, "lat.dat"), np.transpose(lattice))
        logger.info(f"Writing reciprocal lattice to {os.path.join(outDir, 'rlat.dat')}")
        np.savetxt(os.path.join(outDir, "rlat.dat"), np.linalg.inv(lattice) * 2 * np.pi)
        logger.info(f"Writing site positions to {os.path.join(outDir, 'site_positions.dat')}")
        np.savetxt(os.path.join(outDir, "site_positions.dat").format(outDir), np.transpose(cart_coords))
        info = {'nsites' : int(cart_coords.shape[0]), 'isorthogonal': False, 'isspinful': False, 'norbits': int(np.sum(site_norbits)), 'fermi_level': 0.0}
        logger.info(f"Writing info to {os.path.join(outDir, 'info.json')}")
        with open('{}/info.json'.format(outDir), 'w') as info_f:
            json.dump(info, info_f)
    return site_norbits

def tcddrf2deeph(s_mat: np.ndarray,
                 labels: List,
                 baseProvider: BaseProvider,
                 outDir: str = './log_tcddrf2deeph',
                 chi_mat=None,
                 ):
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    site_norbits_dict, orbital_types_dict, element = label2orbital(labels, outDir=outDir)
    write_R_sparse(s_mat, 's', outDir)
    if chi_mat is not None:
        write_R_sparse(chi_mat, 'h', outDir)

    site_norbits = write_sys_info(baseProvider, site_norbits_dict, outDir=outDir)

    overlap_dict = parse_matrix(os.path.join(outDir, 'data-SR-sparse_SPIN0.csr'), element, site_norbits, orbital_types_dict)
    logger.info(f"Writing overlaps to {os.path.join(outDir, 'overlaps.h5')}")
    with h5py.File(os.path.join(outDir, "overlaps.h5"), 'w') as fid:
        for key_str, value in overlap_dict.items():
            fid[key_str] = value
    if chi_mat is not None:
        hamiltonian_dict = parse_matrix(os.path.join(outDir, 'data-HR-sparse_SPIN0.csr'), element, site_norbits, orbital_types_dict)
        logger.info(f"Writing hamiltonians to {os.path.join(outDir, 'hamiltonians.h5')}")
        with h5py.File(os.path.join(outDir, "hamiltonians.h5"), 'w') as fid:
            for key_str, value in hamiltonian_dict.items():
                fid[key_str] = value
    if chi_mat is not None:
        get_rc(outDir, outDir, radius=-1, neighbour_file="hamiltonians.h5")
        get_rh(outDir, outDir)
    return

def deeph2tcddrf(hamiltonian_path: str,
                 outDir: str = './log_deeph2tcddrf',
                 ):
    assert os.path.exists(hamiltonian_path), f"{hamiltonian_path} does not exist"
    folder = pathlib.Path(os.path.abspath(hamiltonian_path)).parent
    element_fname = os.path.join(folder, "element.dat")
    orbital_types_fname = os.path.join(folder, "orbital_types.dat")
    assert os.path.exists(element_fname), f"{element_fname} does not exist"
    assert os.path.exists(orbital_types_fname), f"{orbital_types_fname} does not exist"

    element = np.loadtxt(element_fname, dtype=int)
    site_norbits_dict = {}
    orbital_types_dict = {}
    with open(orbital_types_fname, 'r') as f:
        for idx, line in enumerate(f):
            if element[idx] not in orbital_types_dict:
                ls = [int(num) for num in line.split()]
                orbital_types_dict[element[idx]] = ls
                for l in ls:
                    site_norbits_dict[element[idx]] = site_norbits_dict.get(element[idx], 0) + 2 * l + 1
    site_norbits = []
    for atom in element:
        site_norbits.append(site_norbits_dict[atom])

    hamiltonian_mat = restore_matrix(hamiltonian_path, element, site_norbits, orbital_types_dict)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    fname = os.path.join(outDir, 'QAQ_pred.npy')
    logger.info(f"Writing hamiltonian to {fname}")
    np.save(fname, hamiltonian_mat)
    return hamiltonian_mat

if __name__ == "__main__":
    pass
