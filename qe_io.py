from xml.dom import minidom
import numpy as np
from tqdm import tqdm
import threading
import os
import h5py
import pickle
from mpi4py import MPI
import time

class QERead:

    def __init__(self, comm=None):
        self.xml_data = None
        self.comm = comm

    def parse_QE_XML(self, file_name, storeFolder='./wfc/'):
        """
        analyze QE data-file-schema.xml
            param:
                file_name: str
            return:
                 dict of {'nks', 'kweights', 'nbnd', 'eigen', 'occ', 'fftw'}
        """
        rank = 0
        if not self.comm is None:
            rank = self.comm.Get_rank()
        if rank == 0:
            isExist = os.path.exists(storeFolder)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(storeFolder)
        if not self.comm is None:
            self.comm.Barrier()
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
        if rank == 0:
            with open(storeFolder + '/qe_xml.pickle', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.xml_data = out_dict
        return out_dict 

    def parse_QE_wfc(self, file_name, storeFolder='./wfc/'):
        """
        analyze QE wfc*.dat 
            param:
                file_name: str
            return:
                 dict of {'ik', 'xk', 'ispin', 'nbnd', 'ngw', 'evc'...}
        """
        hdf5 = '.hdf5' in file_name 

        rank, size = 0, 1
        if not self.comm is None:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
        if rank == 0:
            isExist = os.path.exists(storeFolder)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(storeFolder)
        if not self.comm is None:
            self.comm.Barrier()

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

                wfc_dict = {'ik': ik,
                            'xk': xk,
                            'ispin': ispin,
                            'gamma_only': gamma_only,
                            'scalef': scalef,
                            'ngw': ngw,
                            'igwx': igwx,
                            'npol': npol,
                            'nbnd': nbnd,
                            'b': [b1, b2, b3]}

                f.seek(8,1)
                send_data = None
                if rank == 0:
                    evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")
                    for i in range(nbnd):
                        evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
                        f.seek(8, 1)
                    arrs = np.array_split(evc, size, axis=0)
                    raveled = [np.ravel(arr) for arr in arrs]
                    send_data = np.concatenate(raveled)
                ibnd_global = np.arange(nbnd)
                ibnd_loc=  np.array_split(ibnd_global, size)
                count = [len(a) * npol * igwx for a in ibnd_loc]
                displ = np.array([sum(count[:p]) for p in range(size)])

                recvbuf = np.empty((len(ibnd_loc[rank]), npol*igwx), dtype='complex128')
                self.comm.Scatterv([send_data, count, displ, MPI.COMPLEX16], recvbuf, root=0)

                fftw = self.xml_data['fftw']
                fft_grid = np.array(fftw) // 2 + 1
                if rank == 0:
                    pbar = tqdm(desc='store wfc', total=nbnd)
                for index, i in enumerate(ibnd_loc[rank]):
                    evc_g = np.zeros(fft_grid, dtype=np.complex128)
                    evc_g[mill[:, 0], mill[:, 1], mill[:, 2]] = recvbuf[index, :] 
                    if gamma_only:
                        assert(np.all(mill[0, :] == 0))
                        evc_g[-mill[1:, 0], -mill[1:, 1], -mill[1:, 2]] = np.conj(recvbuf[index, 1:]) 
                    evc_r = np.fft.ifftn(evc_g, norm='forward')
                    fileName = storeFolder + '/wfc_' + str(ispin) + '_' + str(ik).zfill(3) + '_' + str(i + 1).zfill(5) + '_r'
                    np.save(fileName, evc_r)
                    if rank == 0:
                        value = np.sum([len(loc) > index for loc in ibnd_loc]) 
                        pbar.update(value)
                if rank == 0:
                    pbar.close()
            return wfc_dict

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
                        evc = evc[:, 0::2] + 1j * evc[:, 1::2]
            return wfc_dict

    def info(self):
        rank = 0
        if not self.comm is None:
            rank = self.comm.Get_rank()
        if rank == 0:
            print("----------------QE XML-------------------")
            print(f"{'cell':^10}:")
            print(self.xml_data['cell'])
            print('\n')
            print(f"{'occupation':^10}:")
            print(self.xml_data['occ'])
            print('\n')
            print(f"{'nbnd':^10}: {self.xml_data['nbnd']:10.5f}")
            print(f"{'nspin':^10}: {self.xml_data['nspin']:10.5f}")
            print(f"{'nks':^10}: {self.xml_data['nks']:10.5f}")
            print(f"{'fermiEne':^10}: {self.xml_data['fermi']:10.5f}")
            print(f"{'npv':^10}:")
            print(self.xml_data['fftw'])
            print('\n')
            print("----------------QBOX XML-------------------")
        return self.xml_data

if __name__ == "__main__":
    # test
    st = time.time()
    comm = MPI.COMM_WORLD
    qe = QERead(comm)
    qe.parse_QE_XML("../bn.save/data-file-schema.xml")
    qe.parse_QE_wfc("../bn.save/wfc1.dat")
    qe.info()
    rank = comm.Get_rank()
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    comm.Barrier()
    if rank == 0:
        print('Execution time:', elapsed_time, 'seconds')
