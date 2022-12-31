from xml.dom import minidom
import numpy as np
import base64, os
import utils
import xml.sax
from lxml import etree

class QBOXRead:
    wfc_data = None

    def __init__(self):
        pass

    def parse_QBOX_XML(self, file_name):
        """
        analyze qbox sample xml files 
            param:
                file_name: str
            return:
                dict of {'nbnd', 'fftw', 'nspin', 'evc'}
        """
        tree = etree.parse(file_name)
        elem = tree.getroot()

        # cell and b
        cell = np.zeros((3, 3))
        b = np.zeros((3, 3))
        cell[0] = [float(num) for num in elem.xpath('//unit_cell/@a')[0].split()]
        cell[1] = [float(num) for num in elem.xpath('//unit_cell/@b')[0].split()]
        cell[2] = [float(num) for num in elem.xpath('//unit_cell/@c')[0].split()]
        volume = abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        fac = 2.0 * np.pi
        b[0] = fac / volume * np.cross(cell[1], cell[2])
        b[1] = fac / volume * np.cross(cell[2], cell[0])
        b[2] = fac / volume * np.cross(cell[0], cell[1])

        # fftw parameter
        fftw = np.zeros(3, dtype=np.int32)
        fftw[0] = int(elem.xpath('//grid/@nx')[0]) 
        fftw[1] = int(elem.xpath('//grid/@ny')[0]) 
        fftw[2] = int(elem.xpath('//grid/@nz')[0]) 

        # nspin and states. kpoint = 1!
        nspin = int(elem.xpath('//wavefunction/@nspin')[0])
        ecut = float(elem.xpath('//wavefunction/@ecut')[0]) * 2.0
        nel = int(elem.xpath('//wavefunction/@nel')[0])
        nempty = int(elem.xpath('//wavefunction/@nempty')[0])

        # wfc and occ
        nbnd = int(elem.xpath('//slater_determinant/@size')[0])
        encoding = elem.xpath('//grid_function/@encoding')[0]
        dtype = np.double
        wfc = np.zeros((nspin, nbnd, *fftw), dtype=dtype)
        occ = np.zeros((nspin, nbnd), dtype=np.int32)
        if nspin == 1:
            occ[nspin - 1, :nel // 2] = 2
            occ[nspin - 1, nel // 2: nel // 2 + nel % 2] = 1
        else:
            # spin up
            occ[0, :(nel + 1) // 2] = 1
            # spin down
            occ[1, :nel // 2] = 1
        wfcs = elem.xpath('//grid_function/text()')
        ispin, iwfc = 0, 0
        for wfc_ in wfcs:
            if encoding.strip() == "text":
                wfc_flatten = np.fromstring(wfc_, dtype=dtype, sep=' ')
                wfc[ispin, iwfc, :, :, :] = wfc_flatten.reshape(fftw)
            else:
                wfc_byte = base64.decodebytes(bytes(wfc_, 'utf-8'))
                wfc_flatten = np.frombuffer(wfc_byte, dtype=dtype)
                wfc[ispin, iwfc, :, :, :] = wfc_flatten.reshape(fftw)
            iwfc = (iwfc + 1) % nbnd
            if iwfc == 0:
                ispin += 1

        # npv
        fac = np.sqrt(4 * ecut) / 2.0 / np.pi
        hmax = int(1.5 + fac * np.linalg.norm(cell[0])) * 2
        kmax = int(1.5 + fac * np.linalg.norm(cell[1])) * 2
        lmax = int(1.5 + fac * np.linalg.norm(cell[2])) * 2
        while utils.factorizable(hmax) is False:
            hmax += 2
        hmax += 2
        while utils.factorizable(kmax) is False:
            kmax += 2
        kmax += 2
        while utils.factorizable(lmax) is False:
            lmax += 2
        lmax += 2

        while utils.factorizable(hmax) is False:
            hmax += 2
        while utils.factorizable(kmax) is False:
            kmax += 2
        while utils.factorizable(lmax) is False:
            lmax += 2
        npv = np.array([hmax, kmax, lmax])

        wfc_dict = {'cell': cell,
                    'b': b,
                    'ecut': ecut,
                    'volume': volume,
                    'nspin': nspin,
                    'nbnd': nbnd,
                    'nel': nel,
                    'nempty': nempty,
                    'evc': wfc,
                    'occ': occ,
                    'fftw': fftw,
                    'npv': npv}
        self.wfc_data = wfc_dict
        return wfc_dict


    def storeWFC(self, realSpace=True, storeFolder='./wfc/'):
        """
        storing wfc in storeFolder 
        param:
            realSpace: bool, whether convert wfc to real space
            storeFolder: str, whether the wfc is strored
        """
        isExist = os.path.exists(storeFolder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(storeFolder)
        nspin = self.wfc_data['nspin']
        nbnd = self.wfc_data['nbnd']
        for ispin in range(nspin):
            for ibnd in range(nbnd):
                if realSpace:
                    fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(ibnd + 1).zfill(5) + '_r'
                    np.save(fileName, self.wfc_data['evc'][ispin, ibnd])
                else:
                    fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(ibnd + 1).zfill(5) + '_g'
                    evc_g = np.fft.fftn(self.wfc_data['evc'][ispin, ibnd], norm='forward')
                    np.save(fileName, evc_g)

    def info(self):
        print("----------------QBOX XML-------------------")
        print(f"{'cell':^10}:")
        print(self.wfc_data['cell'])
        print('\n')
        print(f"{'b':^10}:")
        print(self.wfc_data['b'])
        print('\n')
        print(f"{'volume':^10}: {self.wfc_data['volume']:10.5f}")
        print('\n')
        print(f"{'occupation':^10}:")
        print(self.wfc_data['occ'])
        print('\n')
        print(f"{'nbnd':^10}: {self.wfc_data['nbnd']:10.5f}")
        print('\n')
        print(f"{'nel':^10}: {self.wfc_data['nel']:10.5f}")
        print('\n')
        print(f"{'fftw':^10}:")
        print(self.wfc_data['fftw'])
        print('\n')
        print(f"{'npv':^10}:")
        print(self.wfc_data['npv'])
        print("----------------QBOX XML-------------------")
        return self.wfc_data
        

if __name__ == "__main__":
    # test
    qbox = QBOXRead()
    qbox.parse_QBOX_XML("../gs_b.gs.xml")
    qbox.info()
