from xml.dom import minidom
import numpy as np
from tqdm import tqdm
import threading
import os
import h5py

class QERead:
    storeIbndList = []
    lock_Ibnds = threading.Lock()
    iBnds_condition = threading.Condition(lock_Ibnds)
    xml_data = None
    wfc_data = None
    threadNum = None

    def __init__(self, numThread=15):
        self.threadNum = numThread

    def parse_QE_XML(self, file_name):
        """
        analyze QE data-file-schema.xml
            param:
                file_name: str
            return:
                 dict of {'nks', 'kweights', 'nbnd', 'eigen', 'occ', 'fftw'}
        """
        # unit convert
        hartree2ev = 27.2114

        file = minidom.parse(file_name)
        # cell
        cell = np.zeros((3, 3))
        cell_nodes = file.getElementsByTagName('cell')
        cellTag = ['a1', 'a2', 'a3']
        for index, tag in enumerate(cellTag):
            a_nodes = cell_nodes[0].getElementsByTagName(tag)
            cell[index, :] = [float(num) for num in a_nodes[0].firstChild.data.split()]

        # spin
        spin_nodes = file.getElementsByTagName('spin')
        lsda_nodes = spin_nodes[0].getElementsByTagName('lsda')
        nspin = 1
        if lsda_nodes[0].firstChild.data == 'true':
            nspin = 2

        # num of KS states
        nbnd_up, nbnd_dw, nbnd = -1, -1, -1
        nbnd_nodes = file.getElementsByTagName('nbnd')
        try: 
            nbnd = int(nbnd_nodes[0].firstChild.data)
        except IndexError:
            pass
        if nspin == 2:
            nbnd_up_nodes = file.getElementsByTagName('nbnd_up')
            nbnd_up = int(nbnd_up_nodes[0].firstChild.data)
            nbnd_dw_nodes = file.getElementsByTagName('nbnd_dw')
            nbnd_dw = int(nbnd_dw_nodes[0].firstChild.data)
            if nbnd != -1:
                assert(nbnd == nbnd_dw and nbnd == nbnd_up)
            else:
                assert(nbnd_up == nbnd_dw)
                nbnd = nbnd_dw

        # fermi energy
        fermi_nodes = file.getElementsByTagName('fermi_energy')
        fermiEne = float(fermi_nodes[0].firstChild.data) * hartree2ev

        # num of kpoints
        nks_nodes = file.getElementsByTagName('nks')
        nks = int(nks_nodes[0].firstChild.data)
     
        # [nspin * nks * nbnd]
        eigenvalues, occupations, weights = np.zeros((nspin, nks, nbnd)), np.zeros((nspin, nks, nbnd)), np.zeros(nks)

        ks_nodes = file.getElementsByTagName('ks_energies')
        for index, ks_node in enumerate(ks_nodes):
            # kpoint weight
            k_point_node = ks_node.getElementsByTagName('k_point')
            weights[index] = float(k_point_node[0].attributes['weight'].value)

            # eigenvalues
            eigens_node = ks_node.getElementsByTagName('eigenvalues')
            eigenvalue_ = eigens_node[0].firstChild.data
            eigenvalue_ = [float(num) * hartree2ev for num in eigenvalue_.split()]
            for ispin in range(nspin):
                eigenvalues[ispin, index, :] = eigenvalue_[ispin * nbnd : (ispin + 1) * nbnd]

            # occupation
            # occ_nodes = ks_node.getElementsByTagName('occupations')
            # occupation_ = occ_nodes[0].firstChild.data
            # occupation_ = [float(num) for num in occupation_.split()]
            # for ispin in range(nspin):
            #     occupations[ispin, index, :] = occupation_[ispin * nbnd : (ispin + 1) * nbnd]

            # occupation another way
            occupations[eigenvalues <= fermiEne] = 1
            occupations[eigenvalues > fermiEne] = 0

        # fft grids
        fft_nodes = file.getElementsByTagName('fft_grid')
        np0v = int(fft_nodes[0].attributes['nr1'].value)
        np1v = int(fft_nodes[0].attributes['nr2'].value)
        np2v = int(fft_nodes[0].attributes['nr3'].value)
        out_dict = {'cell': cell,
                    'nspin': nspin,
                    'nks': nks,
                    'fermi': fermiEne,
                    'kweights': weights,
                    'nbnd': nbnd,
                    'eigen': eigenvalues,
                    'occ': occupations,
                    'fftw': [np0v, np1v, np2v]}
        self.xml_data = out_dict
        return out_dict 

    def parse_QE_wfc(self, file_name, hdf5=False):
        """
        analyze QE wfc*.dat 
            param:
                file_name: str
            return:
                 dict of {'ik', 'xk', 'ispin', 'nbnd', 'ngw', 'evc'...}
        """
        wfc_dict = None
        if not hdf5:
            with open(file_name, 'rb') as f:
                # Moves the cursor 4 bytes to the right
                f.seek(4)

                ik = np.fromfile(f, dtype='int32', count=1)[0]
                xk = np.fromfile(f, dtype='float64', count=3)
                ispin = np.fromfile(f, dtype='int32', count=1)[0]
                gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
                scalef = np.fromfile(f, dtype='float64', count=1)[0]

                # Move the cursor 8 byte to the right
                f.seek(8, 1)

                ngw = np.fromfile(f, dtype='int32', count=1)[0]
                igwx = np.fromfile(f, dtype='int32', count=1)[0]
                npol = np.fromfile(f, dtype='int32', count=1)[0]
                nbnd = np.fromfile(f, dtype='int32', count=1)[0]

                # Move the cursor 8 byte to the right
                f.seek(8, 1)

                b1 = np.fromfile(f, dtype='float64', count=3)
                b2 = np.fromfile(f, dtype='float64', count=3)
                b3 = np.fromfile(f, dtype='float64', count=3)

                f.seek(8,1)
                
                mill = np.fromfile(f, dtype='int32', count=3*igwx)
                mill = mill.reshape( (igwx, 3) ) 

                evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")
                f.seek(8,1)
                # for i in tqdm(range(nbnd), desc='read wfc bands'):
                for i in range(nbnd):
                    evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
                    f.seek(8, 1)
                wfc_dict = {'ik': ik,
                            'xk': xk,
                            'ispin': ispin,
                            'gamma_only': gamma_only,
                            'scalef': scalef,
                            'ngw': ngw,
                            'igwx': igwx,
                            'npol': npol,
                            'nbnd': nbnd,
                            'mill': mill,
                            'b': [b1, b2, b3],
                            'evc': evc}
        else:
            wfc_dict = {}
            with h5py.File(file_name, 'r') as f:
                for key, value in f.attrs.items():
                    if key == "gamma_only":
                        value = bool(value)
                    wfc_dict[key] = value
                for key, value in f.items():
                    if key == "MillerIndices":
                        wfc_dict['mill'] = value[()]
                        b1 = value.attrs['bg1']
                        b2 = value.attrs['bg2']
                        b3 = value.attrs['bg3']
                        wfc_dict['b'] = [b1, b2, b3]
                    if key == "evc":
                        evc = value[()]
                        wfc_dict[key] = evc[:, 0::2] + 1j * evc[:, 1::2]

        self.wfc_data = wfc_dict
        return wfc_dict

    def computeWFC(self, realSpace=True, Store=False, storeFolder=None):
        """
        Compute wavefunction in a multi-threaded way
        by either storing wfc in memory or disk
        param:
            xml_data: return of parse_QE_XML
            wfc_data: return of parse_QE_wfc
            realSpace: bool, whether convert wfc to real space
            Store: bool, whether store wfc on disk to save memory
            storeFolder: str, whether the wfc is strored
            threadNum: number of thread to store wfc
        """
        fft_grid = self.xml_data['fftw']
        fft_grid = np.array(fft_grid) // 2 + 1
        if not Store:
            # store everyting in memory
            evc_g = np.zeros([self.wfc_data['nbnd'], fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
            # for index, gvec in enumerate(tqdm(self.wfc_data['mill'], desc='store wfc_g')):
            for index, gvec in enumerate(self.wfc_data['mill']):
                eig = self.wfc_data['evc'][:, index]
                evc_g[:, gvec[0], gvec[1], gvec[2]] = eig
                if self.wfc_data['gamma_only'] is True:
                    if (gvec[0] != 0 or gvec[1] != 0 or gvec[2] != 0):
                        evc_g[:, -gvec[0], -gvec[1], -gvec[2]] = np.conj(eig)
            if not realSpace:
                return evc_g
            else:
                evc_r = np.fft.ifftn(evc_g, axes=(1, 2, 3,), norm='forward')
                return evc_r
        else:
            # memory is not enough, store file under folder for further loading
            assert(storeFolder is not None)
            isExist = os.path.exists(storeFolder)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(storeFolder)
            millIndex = self.wfc_data['mill']
            allZeroIndex = -1
            for index in range(millIndex.shape[0]):
                if millIndex[index, 0] == 0 and millIndex[index, 1] == 0 and millIndex[index, 2] == 0:
                    allZeroIndex = index
                    break
            if allZeroIndex != -1:
                millIndex = np.delete(millIndex, allZeroIndex, axis=0)

            self.lock_Ibnds.acquire()
            self.storeIbndList = list(range(self.wfc_data['nbnd']))
            self.storeIbndList.append(-1)
            self.iBnds_condition.notify_all()
            self.lock_Ibnds.release()
            
            threads = []
            for _ in range(self.threadNum):
                thread_id = threading.Thread(target=self.storeGWorker, args=(millIndex, fft_grid, allZeroIndex, storeFolder, realSpace,))
                threads.append(thread_id)

            for thread_id in threads:
                thread_id.start()

            for thread_id in threads:
                thread_id.join()
            
            # TODO: multithread
            # PERF: Fully Optimized

            # for ibnd in tqdm(range(wfc_data['nbnd']), desc='store ibnd'):
            #     evc_g = np.zeros([fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
            #     wfc_slice = list(wfc_data['evc'][ibnd, :])
            #     evc_g[wfc_data['mill'][:, 0], wfc_data['mill'][:, 1], wfc_data['mill'][:, 2]] = wfc_data['evc'][ibnd, :]
            #     if allZeroIndex != -1:
            #         del wfc_slice[allZeroIndex]
            #     evc_g[ - millIndex[:, 0], - millIndex[:, 1], - millIndex[:, 2]] = np.conj(wfc_slice)
            #     evc_r = None
            #     if not realSpace:
            #         wfcName = 'wfc_g_' + str(ibnd + 1).zfill(5)
            #         np.save(storeFolder + '/' + wfcName, evc_g)
            #     else:
            #         evc_r = np.fft.ifftn(evc_g, norm='forward')
            #         wfcName = 'wfc_r_' + str(ibnd + 1).zfill(5)
            #         np.save(storeFolder + '/' + wfcName, evc_r)
            return

    def storeGWorker(self, millIndex, fft_grid, allZeroIndex, storeFolder, realSpace):
        while True:
            self.lock_Ibnds.acquire()
            while len(self.storeIbndList) == 0:
                self.iBnds_condition.wait()
            iBnd = self.storeIbndList.pop(0)
            # poison
            if iBnd == -1:
                self.storeIbndList.append(-1)
                self.lock_Ibnds.release()
                return
            # not a poison
            self.lock_Ibnds.release()

            # for ibnd in tqdm(iBnds, desc='store ibnd'):
            evc_g = np.zeros([fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
            wfc_slice = self.wfc_data['evc'][iBnd, :]
            evc_g[self.wfc_data['mill'][:, 0], self.wfc_data['mill'][:, 1], self.wfc_data['mill'][:, 2]] = self.wfc_data['evc'][iBnd, :]
            if allZeroIndex != -1:
                wfc_slice = np.delete(wfc_slice, allZeroIndex)
            if self.wfc_data['gamma_only'] is True:
                evc_g[- millIndex[:, 0], - millIndex[:, 1], - millIndex[:, 2]] = np.conj(wfc_slice)
            evc_r = None
            if not realSpace:
                wfcName = 'wfc_g_' + str(iBnd + 1).zfill(5)
                np.save(storeFolder + '/' + wfcName, evc_g)
            else:
                evc_r = np.fft.ifftn(evc_g, norm='forward')
                wfcName = 'wfc_r_' + str(iBnd + 1).zfill(5)
                np.save(storeFolder + '/' + wfcName, evc_r)

    def info(self):
        qe_data = self.xml_data | self.xml_data
        return qe_data

if __name__ == "__main__":
    qe = QERead(15)
    wfc_data = qe.parse_QE_wfc("../8bvo_6feooh_nspin2_hdf5.save/wfcdw1.hdf5", hdf5=True)
    print(wfc_data['mill'].shape)
    print(wfc_data['evc'].shape)
    print(wfc_data['gamma_only'])
