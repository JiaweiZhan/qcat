'''
Take from deepH-pack:
https://github.com/mzjb/DeepH-pack/blob/090e0d5298a4c20c8f395fc3136b410b7a943480/deeph/preprocess/abacus_get_data.py#L38
'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import h5py

class OrbA2B(object):
    def __init__(self, Us_a2b):
        self.Us_a2b = Us_a2b

    def get_U(self, l):
        if l > 3:
            raise NotImplementedError("Only support l = s, p, d, f")
        return self.Us_a2b[l]

    def transform(self, mat, l_lefts, l_rights):
        block_lefts = block_diag(*[self.get_U(l_left) for l_left in l_lefts])
        block_rights = block_diag(*[self.get_U(l_right) for l_right in l_rights])
        return block_lefts @ mat @ block_rights.T

class OrbPYSCF2OpenMX(OrbA2B):
    def __init__(self):
        Us_pyscf2openmx = {}
        Us_pyscf2openmx[0] = np.eye(1)
        Us_pyscf2openmx[1] = np.eye(3)[[0, 1, 2]]
        Us_pyscf2openmx[2] = np.eye(5)[[2, 4, 0, 3, 1]]
        Us_pyscf2openmx[3] = np.eye(7)[[3, 4, 2, 5, 1, 6, 0]]
        super().__init__(Us_pyscf2openmx)

class OrbOpenMX2PYSCF(OrbA2B):
    def __init__(self):
        Us_openmx2pyscf = {}
        Us_openmx2pyscf[0] = np.eye(1)
        Us_openmx2pyscf[1] = np.eye(3)[[0, 1, 2]]
        Us_openmx2pyscf[2] = np.eye(5)[[2, 4, 0, 3, 1]]
        Us_openmx2pyscf[3] = np.eye(7)[[6, 4, 2, 0, 1, 3, 5]]
        super().__init__(Us_openmx2pyscf)

def parse_matrix(matrix_path, element,
                 site_norbits, orbital_types_dict):
    U_orbital = OrbPYSCF2OpenMX()
    site_norbits_cumsum = np.cumsum(site_norbits)
    nsites = len(site_norbits)
    matrix_dict = dict()
    with open(matrix_path, 'r') as f:
        line = f.readline() # read "Matrix Dimension of ..."
        if not "Matrix Dimension of" in line:
            line = f.readline() # ABACUS >= 3.0
            assert "Matrix Dimension of" in line
        f.readline() # read "Matrix number of ..."
        norbits = int(line.split()[-1])
        for line in f:
            line1 = line.split()
            if len(line1) == 0:
                break
            num_element = int(line1[3])
            if num_element != 0:
                R_cur = np.array(line1[:3]).astype(int)
                line2 = f.readline().split()
                line3 = f.readline().split()
                line4 = f.readline().split()
                hamiltonian_cur = csr_matrix((np.array(line2).astype(float), np.array(line3).astype(int),
                                                np.array(line4).astype(int)), shape=(norbits, norbits)).toarray()
                for index_site_i in range(nsites):
                    for index_site_j in range(nsites):
                        key_str = f"[{R_cur[0]}, {R_cur[1]}, {R_cur[2]}, {index_site_i + 1}, {index_site_j + 1}]"
                        mat = hamiltonian_cur[(site_norbits_cumsum[index_site_i] - site_norbits[index_site_i]):site_norbits_cumsum[index_site_i],
                                            (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]): site_norbits_cumsum[index_site_j]]
                        if abs(mat).max() < 1e-8:
                            continue
                        mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]],
                                                    orbital_types_dict[element[index_site_j]])
                        matrix_dict[key_str] = mat
    return matrix_dict

def restore_matrix(hamiltonian_path, element,
                   site_norbits, orbital_types_dict):
    U_orbital = OrbOpenMX2PYSCF()
    site_norbits_cumsum = np.cumsum(site_norbits)
    nsites = len(site_norbits)
    matrix_dict = dict()
    with h5py.File(hamiltonian_path, 'r') as f:
        for key in f.keys():
            data = f[key][:]
            matrix_dict[key] = data

    hamiltonian = np.zeros((site_norbits_cumsum[-1], site_norbits_cumsum[-1]), dtype=float)
    for index_site_i in range(nsites):
        for index_site_j in range(nsites):
            key_str = f"[0, 0, 0, {index_site_i + 1}, {index_site_j + 1}]"
            mat = matrix_dict[key_str]
            mat = U_orbital.transform(mat, orbital_types_dict[element[index_site_i]],
                                        orbital_types_dict[element[index_site_j]])
            hamiltonian[(site_norbits_cumsum[index_site_i] - site_norbits[index_site_i]):
                    site_norbits_cumsum[index_site_i],
                    (site_norbits_cumsum[index_site_j] - site_norbits[index_site_j]):
                    site_norbits_cumsum[index_site_j]] = mat
    hamiltonian = (hamiltonian + hamiltonian.T) / 2
    return hamiltonian
