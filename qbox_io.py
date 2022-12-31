from xml.dom import minidom
import numpy as np
import base64, os
import utils

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
        file = minidom.parse(file_name)

        # cell parameter
        cell_nodes = file.getElementsByTagName('unit_cell')
        cellTag = ['a', 'b', 'c']
        cell = np.zeros((3, 3))
        b = np.zeros((3, 3))
        for index, tag in enumerate(cellTag):
            cell[index, :] = [float(num) for num in cell_nodes[0].attributes[tag].value.split()]
        volume = abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        fac = 2.0 * np.pi
        b[0] = fac / volume * np.cross(cell[1], cell[2])
        b[1] = fac / volume * np.cross(cell[2], cell[0])
        b[2] = fac / volume * np.cross(cell[0], cell[1])

        # fftw parameter
        grid_nodes = file.getElementsByTagName('grid')
        fftwTag = ['nx', 'ny', 'nz']
        fftw = np.zeros(3, dtype=np.int32)
        for index, tag in enumerate(fftwTag):
            fftw[index] =  float(grid_nodes[0].attributes[tag].value)

        # nspin and states. kpoint = 1!
        wfc_nodes = file.getElementsByTagName('wavefunction')
        nspin = int(wfc_nodes[0].attributes['nspin'].value)
        ecut = float(wfc_nodes[0].attributes['ecut'].value) * 2
        nel = int(wfc_nodes[0].attributes['nel'].value)
        nempty = int(wfc_nodes[0].attributes['nempty'].value)

        # read wfc
        slater_nodes = file.getElementsByTagName('slater_determinant')
        nbnd = int(slater_nodes[0].attributes['size'].value)
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
        for ispin, node in enumerate(slater_nodes):
            wfc_nodes = node.getElementsByTagName('grid_function')
            for iwfc, wfc_node in enumerate(wfc_nodes):
                encoding = wfc_node.attributes['encoding'].value
                # wfc_r
                if encoding.strip() == "text":
                    wfc_flatten = np.fromstring(wfc_node.firstChild.data, dtype=dtype, sep=' ')
                    wfc[ispin, iwfc, :, :, :] = wfc_flatten.reshape(fftw)
                else:
                    wfc_byte = base64.decodebytes(bytes(wfc_node.firstChild.data, 'utf-8'))
                    wfc_flatten = np.frombuffer(wfc_byte, dtype=dtype)
                    wfc[ispin, iwfc, :, :, :] = wfc_flatten.reshape(fftw)
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
                    'npv': np.array([hmax, kmax, lmax])}
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
        

if __name__ == "__main__":
    # test
    qbox = QBOXRead()
    wfc_data = qbox.parse_QBOX_XML('../gs_b.gs.xml')
    print(wfc_data['cell'])
    print(wfc_data['b'])
    print(wfc_data['volume'])
    print(wfc_data['occ'])
    print(wfc_data['nbnd'])
    print(wfc_data['nel'])
    print(wfc_data['npv'])
