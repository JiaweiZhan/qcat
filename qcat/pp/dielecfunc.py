from lxml import etree
import numpy as np
from ..utils.gpu_kernels import gaussian3d

class DielecFunc(object):
    def __init__(self,
                 xml_fname=None,
                 ff_amplitude=0.0,
                 unit_cell=np.zeros((3, 3)),
                 npv=np.zeros(3, dtype=np.int32),
                 nspin=1,
                 e_field=np.array([]),
                 mlwf_center={},
                 mlwf_occupation={},
                 mlwf_spread={},
                 ):
        self.xml_fname = xml_fname
        self.ff_amplitude = ff_amplitude
        self.unit_cell = unit_cell # unit_cell.shape = (3, 3)
        self.npv = npv
        self.nspin = nspin
        self.e_field = e_field                   # e_field.shape = (nefield, 3)
        self.mlwf_center = mlwf_center           # mlwf_center["0"].shape = (nefield, nmlwf_up, 3)
        self.mlwf_occupation = mlwf_occupation   # mlwf_occupation["0"].shape = (nefield, nmlwf_up, 1)
        self.mlwf_spread = mlwf_spread           # mlwf_spread["0"].shape = (nefield, nmlwf_up, 1)
        self.center = {}                         # center["0"].shape = (nefield // 2, nmlwf_up, 3) in relative scale
        self.dspl = {}                           # dspl["0"].shape = (nefield // 2, nmlwf_up, 3)   in bohr unit
        self.spread = {}                         # spread["0"].shape = (nefield // 2, nmlwf_up, 1) in bohr unit
        self.polarization = {}                   # polarization["0"].shape = (nefield // 2, npv[0], npv[1], npv[2])

        if xml_fname:
            self.parse_xml()

    def parse_xml(self):
        context = etree.iterparse(self.xml_fname, huge_tree=True)
        e_fields = []
        mlwf_centers = {0: [], 1: []}
        mlwf_spreads = {0: [], 1: []}
        mlwf_occupations = {0: [], 1: []}
        for _, element in context:

            if element.tag == 'cmd':
                cmd = element.text
                if cmd and 'response' in cmd:
                    self.ff_amplitude = float(cmd.strip().split()[1])

            if element.tag == 'np0v':
                self.npv[0] = int(element.text)
            if element.tag == 'np1v':
                self.npv[1] = int(element.text)
            if element.tag == 'np2v':
                self.npv[2] = int(element.text)

            if element.tag == 'unit_cell':
                self.unit_cell[0] = np.array([float(num) for num in element.attrib['a'].split()])
                self.unit_cell[1] = np.array([float(num) for num in element.attrib['b'].split()])
                self.unit_cell[2] = np.array([float(num) for num in element.attrib['c'].split()])

            if element.tag == 'wavefunction':
                self.nspin = int(element.attrib['nspin'])

            if element.tag == 'e_field':
                e_field = np.array([float(num) for num in element.text.split()])
                if not np.equal(e_field, 0.0).all():
                    e_fields.append(e_field)

            if element.tag == 'mlwf_set':
                ispin = int(element.attrib['spin'])
                if ispin == 1:
                    self.nspin = 2
                mlwf_center = []
                mlwf_spread = []
                mlwf_occupation = []

                for subchild in element:
                    if subchild.tag == 'mlwf':
                        spread = [float(subchild.attrib['spread'])]
                        mlwf_spread.append(spread)

                        occupation = [float(subchild.attrib['occupation'])]
                        mlwf_occupation.append(occupation)
                    if subchild.tag == 'mlwf_ref':
                        center = [float(num) for num in subchild.attrib['center'].strip().split()]
                        mlwf_center.append(center)

                mlwf_spreads[ispin].append(np.stack(mlwf_spread, axis=0))
                mlwf_occupations[ispin].append(np.stack(mlwf_occupation, axis=0))
                mlwf_centers[ispin].append(np.stack(mlwf_center, axis=0))
        for ispin in range(self.nspin):
            self.mlwf_center[ispin] = np.stack(mlwf_centers[ispin], axis=0)
            self.mlwf_spread[ispin] = np.stack(mlwf_spreads[ispin], axis=0)
            self.mlwf_occupation[ispin] = np.stack(mlwf_occupations[ispin], axis=0)

        self.e_field = np.stack(e_fields, axis=0)

    def MLWFCenterDspl(self):
        '''
        locate the mlwf by finding the mlwf with closest centers between two opposite finite fields.
        '''
        nefield = self.e_field.shape[0]
        nspin = self.nspin
        centers = {0: [], 1: []}
        dspls = {0: [], 1: []}
        spreads = {0: [], 1: []}
        for ispin in range(nspin):
            for iefield in range(0, nefield, 2):
                center_1 = self.mlwf_center[ispin][iefield]       # (nmlwf, 3)
                center_2 = self.mlwf_center[ispin][iefield + 1]   # (nmlwf, 3)
                spread_1 = self.mlwf_spread[ispin][iefield]       # (nmlwf, 1)
                spread_2 = self.mlwf_spread[ispin][iefield + 1]   # (nmlwf, 1)
                center_1_rel = center_1 @ np.linalg.inv(self.unit_cell)
                center_2_rel = center_2 @ np.linalg.inv(self.unit_cell)
                center_diff = - center_1_rel[:, np.newaxis, :] % 1 + center_2_rel[np.newaxis, :, :] % 1
                center_diff = (center_diff + 0.5) % 1 - 0.5
                center_diff_norm = np.linalg.norm(center_diff @ self.unit_cell, axis=2)
                mlwf_index = np.argmin(center_diff_norm, axis=1)
                assert np.all(np.sort(mlwf_index) == np.arange(mlwf_index.shape[0]))

                dspl_rel = center_diff[np.arange(center_diff.shape[0]), mlwf_index] * self.mlwf_occupation[ispin][iefield]
                dspl = dspl_rel @ self.unit_cell
                dspls[ispin].append(dspl)

                center = (center_1_rel + 0.5 * dspl_rel) % 1
                centers[ispin].append(center)

                spread = 0.5 * spread_1 + 0.5 * spread_2[mlwf_index]
                spreads[ispin].append(spread)
            self.center[ispin] = np.stack(centers[ispin], axis=0)
            self.dspl[ispin] = np.stack(dspls[ispin], axis=0)
            self.spread[ispin] = np.stack(spreads[ispin], axis=0)

    @staticmethod
    def gaussian3d(unit_cell,      # [3, 3]
                   r1: np.ndarray, # [ngrid, 3] in relative scale
                   r2: np.ndarray, # [nefield // 2, nmlwf, 3] in relative scale
                   spread: np.ndarray, # [nefield //2, nmlwf, 1] in bohr unit
                   dspl_norm: np.ndarray, # [nefield // 2, nmlwf, 1] in bohr unit
                   ):
        '''
        return shape: [nefield // 2, ngrid]
        FIXME: this function is not optimized, GPU acceleration per grid is possible.
        '''
        return gaussian3d(unit_cell, r1, r2, spread, dspl_norm)

    def computeLocalPolarization(self,
                                 spread_factor:float=1.0):
        self.MLWFCenterDspl()
        nspin = self.nspin
        i, j, k = np.indices((self.npv[0], self.npv[1], self.npv[2]), dtype=np.float64)
        i /= self.npv[0]
        j /= self.npv[1]
        k /= self.npv[2]
        r1 = np.stack([i, j, k], axis=-1)
        r1 = r1.reshape(-1, 3)
        mask = np.where(self.e_field != 0.0, 1.0, 0.0)  # [nefield, 3]
        for ispin in range(nspin):
            center = self.center[ispin]
            spread = self.spread[ispin]
            dspl_norm = np.sum(self.dspl[ispin] * mask[::2][:, None, :], axis=-1, keepdims=True)
            self.polarization[ispin] =  self.gaussian3d(self.unit_cell, r1, center, spread * spread_factor, dspl_norm).reshape((-1, *self.npv)) * -1.0

    def computeDielecFunc(self,
                          spread_factor:float=1.0):
        self.computeLocalPolarization(spread_factor)
        nspin = self.nspin
        nefield = self.e_field.shape[0]
        eps = {}
        eps_e = {0: [], 1: []}
        for ispin in range(nspin):
            for iefield in range(nefield // 2):
                deltaE = self.e_field[2 * iefield + 1] - self.e_field[2 * iefield]
                deltaE = np.sum(np.sign(deltaE)) * np.linalg.norm(deltaE)
                eps_i = 1.0 + 4 * np.pi * self.polarization[ispin][iefield] / deltaE
                eps_e[ispin].append(eps_i)
            eps[ispin] = np.mean(np.stack(eps_e[ispin], axis=0), axis=0)
        if nspin == 1:
            return eps[0]
        else:
            return eps[0] + eps[1] - 1.0

    def write2qbox(self,
                   eps: np.ndarray,
                   fname: str = 'alpha.txt'):
        with open(fname, 'w') as f:
            for k in range(self.npv[2]):
                for i in range(self.npv[0]):
                    for j in range(self.npv[1]):
                        f.write(f"{eps[i, j, k]:15.8f}")
                f.write('\n')


    def __str__(self):
        return f"npv:\n{self.npv}\n\nff_amplitude:\n{self.ff_amplitude}\n\nunit_cell:\n{self.unit_cell}\n\n\
            nspin:\n{self.nspin}\n\ne_field:\n{self.e_field}\n\n\
            mlwf_center:\n{self.mlwf_center}\n\nmlwf_spread:\n{self.mlwf_spread}\n\n\
            center:\n{self.center}\n\ndspl:\n{self.dspl}\n\nspread:\n{self.spread}"
