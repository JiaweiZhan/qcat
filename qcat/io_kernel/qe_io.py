from xml.dom import minidom
import numpy as np
from tqdm import tqdm
import os
import h5py
import pickle
from mpi4py import MPI
import time
import pathlib
import shutil
from .base_io import Read

class QERead(Read):

    def __init__(self,
                 outFolder: str,
                 comm=None,
                 ):
        self.xml_data = None
        self.comm = comm
        self.qe_outputFolder = outFolder
        assert os.path.exists(outFolder), f"qe output folder {outFolder} does not exist"

    def parse_info(self,
                   store: bool=True,
                   storeFolder: str='./wfc/'):
        """
        analyze QE data-file-schema.xml
            param:
                file_name: str
            return:
                 dict of {'nks', 'kweights', 'nbnd', 'eigen', 'occ', 'fftw'}
        """
        saveFolder = self.qe_outputFolder
        rank = 0
        if self.comm:
            rank = self.comm.Get_rank()
        if rank == 0:
            if store:
                isExist = os.path.exists(storeFolder)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(storeFolder)
        if self.comm:
            self.comm.Barrier()
        # unit convert
        hartree2ev = 27.2114

        #get data-file-schema.xml
        file_name = [saveFolder + '/' + file for file in os.listdir(saveFolder) if 'data-file-schema' in file][0]

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
        nbnd = []
        try: 
            nbnd = [int(nbnd_nodes[0].firstChild.data)]
        except IndexError:
            pass
        if nspin == 2:
            nbnd_up_nodes = file.getElementsByTagName('nbnd_up')
            nbnd_up = int(nbnd_up_nodes[0].firstChild.data)
            nbnd_dw_nodes = file.getElementsByTagName('nbnd_dw')
            nbnd_dw = int(nbnd_dw_nodes[0].firstChild.data)
            # if nbnd != -1:
            #     assert(nbnd == nbnd_dw and nbnd == nbnd_up)
            # else:
            #     assert(nbnd_up == nbnd_dw)
            #     nbnd = [nbnd_dw] * 2
            nbnd = [nbnd_up, nbnd_dw]

        # fermi energy
        fermi_nodes = file.getElementsByTagName('fermi_energy')
        try:
            fermiEne = float(fermi_nodes[0].firstChild.data) * hartree2ev
        except IndexError:
            fermiEne = 0.0

        # atomic positions 
        species_loc = []
        atomic_nodes = file.getElementsByTagName('atomic_positions')
        atom_nodes = atomic_nodes[-1].getElementsByTagName('atom')
        for atom_node in atom_nodes:
            species = atom_node.attributes['name'].value
            loc = [float(num) for num in atom_node.firstChild.data.split()]
            species_loc.append([species] + loc)

        # num of kpoints
        nks_nodes = file.getElementsByTagName('nks')
        nks = int(nks_nodes[0].firstChild.data)
     
        # [nspin * nks * nbnd]
        eigenvalues, occupations, weights = np.zeros((nspin, nks, nbnd[0])), np.zeros((nspin, nks, nbnd[0])), np.zeros(nks)

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
                eigenvalues[ispin, index, :] = eigenvalue_[int(np.sum(nbnd[:ispin])) : int(np.sum(nbnd[:ispin + 1]))]

            # occupation
            occ_nodes = ks_node.getElementsByTagName('occupations')
            occupation_ = occ_nodes[0].firstChild.data
            occupation_ = [float(num) for num in occupation_.split()]
            for ispin in range(nspin):
                occupations[ispin, index, :] = occupation_[int(np.sum(nbnd[:ispin])) : int(np.sum(nbnd[:ispin + 1]))]

            # occupation another way
            # occupations[eigenvalues <= fermiEne] = 1
            # occupations[eigenvalues > fermiEne] = 0

        # fft grids
        fft_nodes = file.getElementsByTagName('fft_grid')
        np0v = int(fft_nodes[0].attributes['nr1'].value)
        np1v = int(fft_nodes[0].attributes['nr2'].value)
        np2v = int(fft_nodes[0].attributes['nr3'].value)
        fftw = np.array([np0v, np1v, np2v]) // 2 + 1 
        out_dict = {'cell': cell,
                    "atompos": species_loc,
                    'nspin': nspin,
                    'nks': nks,
                    'fermi': fermiEne,
                    'kweights': weights,
                    'nbnd': nbnd,
                    'eigen': eigenvalues,
                    'occ': occupations,
                    'npv': [np0v, np1v, np2v],
                    'fftw': fftw}
        if rank == 0:
            if store:
                with open(storeFolder + '/info.pickle', 'wb') as handle:
                    pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.xml_data = out_dict
        return out_dict 

    def parse_wfc(self,
                  real_space=False,
                  storeFolder='./wfc/'):
        """
        analyze QE wfc*.dat 
            param:
                file_name: str
            return:
                 dict of {'ik', 'xk', 'ispin', 'nbnd', 'ngw', 'evc'...}
        """
        rank, size = 0, 1
        if self.comm:
            size = self.comm.Get_size()
            rank = self.comm.Get_rank()
        if rank == 0:
            isExist = os.path.exists(storeFolder)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(storeFolder)
        if self.comm:
            self.comm.Barrier()

        saveFolder = self.qe_outputFolder
        wfc_files = [saveFolder + '/' + file for file in os.listdir(saveFolder) if 'wfc' in file]
        for file_name in wfc_files:
            hdf5 = (".hdf5" == pathlib.Path(file_name).suffix)

            wfc_dict = None
            if not hdf5:
                send_data = None
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
                    if rank == 0:
                        fileName = os.path.join(storeFolder, 'mill')
                        np.save(fileName, mill)

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

                    if rank == 0:
                        evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")
                        for i in range(nbnd):
                            evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
                            f.seek(8, 1)
                    # step out of files
                if rank == 0:
                    arrs = np.array_split(evc, size, axis=0)
                    raveled = [np.ravel(arr) for arr in arrs]
                    send_data = np.concatenate(raveled)
                ibnd_global = np.arange(nbnd)
                ibnd_loc=  np.array_split(ibnd_global, size)
                count = [len(a) * npol * igwx for a in ibnd_loc]
                displ = np.array([sum(count[:p]) for p in range(size)])

                if self.comm:
                    recvbuf = np.empty((len(ibnd_loc[rank]), npol*igwx), dtype='complex128')
                    self.comm.Scatterv([send_data, count, displ, MPI.COMPLEX16], recvbuf, root=0)
                else:
                    recvbuf = evc

                fft_grid = self.xml_data['fftw']
                if rank == 0:
                    pbar = tqdm(desc=f"store wfc of spin:{ispin:^3}/{self.xml_data['nspin']:^3}; ik:{ik:^3}/{self.xml_data['nks']:^3}", total=nbnd)
                for index, i in enumerate(ibnd_loc[rank]):
                    if real_space:
                        evc_g = np.zeros(fft_grid, dtype=np.complex128)
                        evc_g[mill[:, 0], mill[:, 1], mill[:, 2]] = recvbuf[index, :] 
                        if gamma_only:
                            assert(np.all(mill[0, :] == 0))
                            evc_g[-mill[1:, 0], -mill[1:, 1], -mill[1:, 2]] = np.conj(recvbuf[index, 1:]) 
                        data2store = np.fft.ifftn(evc_g, norm='forward')
                        fileName = storeFolder + '/wfc_' + str(ispin) + '_' + str(ik).zfill(3) + '_' + str(i + 1).zfill(5) + '_r'
                    else:
                        data2store = recvbuf[index, :]
                        fileName = storeFolder + '/wfc_' + str(ispin) + '_' + str(ik).zfill(3) + '_' + str(i + 1).zfill(5) + '_g'
                    np.save(fileName, data2store)
                    if rank == 0:
                        value = np.sum([len(loc) > index for loc in ibnd_loc]) 
                        pbar.update(value)
                if rank == 0:
                    pbar.close()

            else:
                if rank == 0:
                    wfc_dict = {}
                    evc = None
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
                    arrs = np.array_split(evc, size, axis=0)
                    raveled = [np.ravel(arr) for arr in arrs]
                    send_data = np.concatenate(raveled)
                else:
                    wfc_dict = None
                    send_data = None
                
                if self.comm:
                    wfc_dict = self.comm.bcast(wfc_dict, root=0)

                if rank == 0:
                    fileName = os.path.join(storeFolder, 'mill')
                    np.save(fileName, np.asarray(wfc_dict['mill']))

                ibnd_global = np.arange(wfc_dict['nbnd'])
                ibnd_loc=  np.array_split(ibnd_global, size)
                count = [len(a) * wfc_dict['npol'] * wfc_dict['igwx'] for a in ibnd_loc]
                displ = np.array([sum(count[:p]) for p in range(size)])

                if self.comm:
                    recvbuf = np.empty((len(ibnd_loc[rank]), wfc_dict['npol'] * wfc_dict['igwx']), dtype='complex128')
                    self.comm.Scatterv([send_data, count, displ, MPI.COMPLEX16], recvbuf, root=0)
                else:
                    recvbuf = evc

                fft_grid = self.xml_data['fftw']
                if rank == 0:
                    pbar = tqdm(desc=f"store wfc of ispin:{wfc_dict['ispin']:^3}/{self.xml_data['nspin']:^3}; ik:{wfc_dict['ik']:^3}/{self.xml_data['nks']:^3}", total=wfc_dict['nbnd'])
                for index, i in enumerate(ibnd_loc[rank]):
                    if real_space:
                        evc_g = np.zeros(fft_grid, dtype=np.complex128)
                        evc_g[wfc_dict['mill'][:, 0], wfc_dict['mill'][:, 1], wfc_dict['mill'][:, 2]] = recvbuf[index, :] 
                        if wfc_dict['gamma_only']:
                            assert(np.all(wfc_dict['mill'][0, :] == 0))
                            evc_g[-wfc_dict['mill'][1:, 0], -wfc_dict['mill'][1:, 1], -wfc_dict['mill'][1:, 2]] = np.conj(recvbuf[index, 1:]) 
                        data2store = np.fft.ifftn(evc_g, norm='forward')
                        fileName = storeFolder + '/wfc_' + str(wfc_dict['ispin']) + '_' + str(wfc_dict['ik']).zfill(3) + '_' + str(i + 1).zfill(5) + '_r'
                    else:
                        data2store = recvbuf[index, :]
                        fileName = storeFolder + '/wfc_' + str(wfc_dict['ispin']) + '_' + str(wfc_dict['ik']).zfill(3) + '_' + str(i + 1).zfill(5) + '_g'
                    np.save(fileName, data2store)
                    if rank == 0:
                        value = np.sum([len(loc) > index for loc in ibnd_loc]) 
                        pbar.update(value)
                if rank == 0:
                    pbar.close()
        fileNameList_tot = [] 
        for isp in range(self.xml_data['nspin']):
            fileNameList_sp = [] 
            for ik in range(self.xml_data['nks']):
                fileNameList_ik = [] 
                for iwf in range(self.xml_data['nbnd'][isp]):
                    fileName = storeFolder + '/wfc_' + str(isp + 1) + '_' + str(ik + 1).zfill(3) + '_' + str(iwf + 1).zfill(5) + '_r.npy'
                    fileNameList_ik.append(fileName)
                fileNameList_sp.append(fileNameList_ik)
            fileNameList_tot.append(fileNameList_sp)
        fileNameList_tot = np.array(fileNameList_tot)

        if rank == 0:
            with open(storeFolder + '/info.pickle', 'rb') as handle:
                info_data = pickle.load(handle)
            info_data['wfc_file'] = fileNameList_tot

            with open(storeFolder + '/info.pickle', 'wb') as handle:
                pickle.dump(info_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.comm:
            self.comm.Barrier()

    def read(self,
             storeFolder: str,
             real_space: bool=True,
             store_wfc: bool=True,
             ):
        self.parse_info(store=True,
                        storeFolder=storeFolder)
        self.parse_wfc(storeFolder=storeFolder,
                       real_space=real_space)

    def info(self):
        rank = 0
        if self.comm:
            rank = self.comm.Get_rank()
        if rank == 0:
            print("----------------QE XML-------------------")
            print(f"{'cell':^10}:")
            print(self.xml_data['cell'])
            print('\n')
            print(f"{'occupation':^10}:")
            print(self.xml_data['occ'])
            print('\n')
            print(f"{'nbnd':^10}: {self.xml_data['nbnd']}")
            print(f"{'nspin':^10}: {self.xml_data['nspin']:10.5f}")
            print(f"{'nks':^10}: {self.xml_data['nks']:10.5f}")
            print(f"{'fermiEne':^10}: {self.xml_data['fermi']:10.5f}")
            print(f"{'fftw':^10}:")
            print(self.xml_data['fftw'])
            print(f"{'npv':^10}:")
            print(self.xml_data['npv'])
            print('\n')
            print("----------------QBOX XML-------------------")
        return self.xml_data

    def clean_wfc(self, storeFolder='./wfc/'):
        rank = 0
        if self.comm:
            rank = self.comm.Get_rank()
        if rank == 0:
            isExist = os.path.exists(storeFolder)
            if isExist:
                shutil.rmtree(storeFolder)
        if self.comm:
            self.comm.Barrier()

if __name__ == "__main__":
    # test
    st = time.time()
    comm = MPI.COMM_WORLD
    qe = QERead(comm=comm, outFolder="/project/gagalli/jiaweiz/test/TEST_WEST/tutorials/h-bn/vacuum_extrapolate/vacuum_20/hbn.save/")
    qe.parse_info()
    # qe.parse_wfc("../8bvo_6feooh_nspin2_hdf5.save")
    qe.info()
    rank = comm.Get_rank()
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    comm.Barrier()
    if rank == 0:
        print('Execution time:', elapsed_time, 'seconds')
