import os
import shutil
import unittest
from qcat.io_kernel.deeph.deeph_interface import label2orbital, tcddrf2deeph, deeph2tcddrf

class abacusIOUnitTest(unittest.TestCase):
    def test_label2orbital(self):
        labels = [
                "3 Si 2px   ", "3 Si 2py   ", "3 Si 2pz   ", "3 Si 3px   ", "3 Si 3py   ", "3 Si 3pz   ", "3 Si 4px   ", "3 Si 4py   ", "3 Si 4pz",
                "4 H 2px   ", "4 H 2py   ", "4 H 2pz   ", "4 H 3px   ", "4 H 3py   ", "4 H 3pz   ", "4 H 4px   ", "4 H 4py   ",
                "4 H 4pz   ", "4 H 5px   ", "4 H 5py   ", "4 H 5pz   ", "4 H 3dxy  ", "4 H 3dyz  ", "4 H 3dz^2 ", "4 H 3dxz  ",
                "4 H 3dx2-y2", "4 H 4dxy  ", "4 H 4dyz  ", "4 H 4dz^2 ", "4 H 4dxz  ", "4 H 4dx2-y2", "4 H 5dxy  ", "4 H 5dyz  ",
                "4 H 5dz^2 ", "4 H 5dxz  ", "4 H 5dx2-y2", "4 H 6dxy  ", "4 H 6dyz  ", "4 H 6dz^2 ", "4 H 6dxz  ", "4 H 6dx2-y2", "4 H 4f-3  ",
                "4 H 4f-2  ", "4 H 4f-1  ", "4 H 4f+0  ", "4 H 4f+1  ", "4 H 4f+2  ", "4 H 4f+3  ", "4 H 5f-3  ", "4 H 5f-2  ", "4 H 5f-1  ",
                "4 H 5f+0  ", "4 H 5f+1  ", "4 H 5f+2  ", "4 H 5f+3  ", "4 H 6f-3  ", "4 H 6f-2  ", "4 H 6f-1  ", "4 H 6f+0  ", "4 H 6f+1  ",
                "4 H 6f+2  ", "4 H 6f+3  ",
                "5 Si 2px   ", "5 Si 2py   ", "5 Si 2pz   ", "5 Si 3px   ", "5 Si 3py   ", "5 Si 3pz   ", "5 Si 4px   ", "5 Si 4py   ", "5 Si 4pz",
        ]
        site_norbits_dict, orbital_types_dict, element = label2orbital(labels, outDir='./log')
        site_norbits_dict_ref = {1: 53, 14: 9}
        orbital_types_dict_ref = {1: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], 14: [1, 1, 1]}
        element_ref = [14, 1, 14]
        self.assertDictEqual(site_norbits_dict, site_norbits_dict_ref)
        self.assertDictEqual(orbital_types_dict, orbital_types_dict_ref)
        self.assertListEqual(element, element_ref)

    def test_parse2restore(self):
        from pyscf.pbc import gto
        from qcat.io_kernel import PYSCFProvider
        import numpy as np
        cell = gto.Cell()
        cell.atom = '''H  0 0 0; H 1 1 1; O 0 1 0'''
        cell.basis = 'cc-pvtz'
        cell.a = np.eye(3) * 2
        cell.build()

        s_mat = np.asarray(cell.pbc_intor('int1e_ovlp_sph'))
        provider = PYSCFProvider(cell)
        outDir = './log'
        tcddrf2deeph(s_mat=s_mat, labels=cell.spheric_labels(), baseProvider=provider, outDir=outDir)
        fname = os.path.join(outDir, "overlaps.h5")
        s_mat_restore = deeph2tcddrf(hamiltonian_path=fname, outDir=outDir)
        shutil.rmtree(outDir)
        self.assertTrue(np.allclose(s_mat, s_mat_restore))

if __name__ == '__main__':
    unittest.main()
