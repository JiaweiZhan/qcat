#!/scratch/midway2/jiaweiz/anaconda3/bin/python3
import argparse
import numpy as np
import base64, os, time
from . import utils
from .io import Read
from lxml import etree
from mpi4py import MPI
import pickle
from mpi4py import MPI
from tqdm import tqdm

class QBOXRead(Read):

    def __init__(self, comm=None):
        self.wfc_data = None
        self.comm = comm

    def parse_info(self, file_name=None, store=True, storeFolder='./wfc/'):
        pass

    def parse_wfc(self, file_name, storeFolder='./wfc/'):
        """
        analyze qbox sample xml files 
            param:
                file_name: str
            return:
                dict of {'nbnd', 'fftw', 'nspin', 'evc'}
        """
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

        context = etree.iterparse(file_name, huge_tree=True, events=('start', 'end'))

        ispin, iwfc = 0, 0
        cell = np.zeros((3, 3))
        b = np.zeros((3, 3))
        fftw = np.zeros(3, dtype=np.int32)
        volume = 0
        nspin, ecut, nel, nempty, nbnd = 0, 0, 0, 0, None
        occ = None
        encoding = "text"

        # several necessary configuration
        for event, element in context:
            if element.tag == "grid_function":
                break
            if element.tag == "unit_cell" and event == 'start':
                cell[0] = [float(num) for num in element.get("a").split()]
                cell[1] = [float(num) for num in element.get("b").split()]
                cell[2] = [float(num) for num in element.get("c").split()]
                volume = abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
                fac = 2.0 * np.pi
                b[0] = fac / volume * np.cross(cell[1], cell[2])
                b[1] = fac / volume * np.cross(cell[2], cell[0])
                b[2] = fac / volume * np.cross(cell[0], cell[1])
            elif element.tag == "grid" and event == 'start':
                fftw[0] = int(element.get("nx")) 
                fftw[1] = int(element.get("ny")) 
                fftw[2] = int(element.get("nz")) 
            elif element.tag == "wavefunction" and event == 'start':
                nspin = int(element.get("nspin"))
                ecut = float(element.get("ecut")) * 2.0
                nel = int(element.get("nel"))
                nempty = int(element.get("nempty"))

                nbnd = np.zeros(nspin, dtype=np.int32)
                if nspin == 1:
                    nbnd[0] = (nel + 1) // 2 + nempty
                else:
                    nbnd[0] = (nel + 1) // 2 + nempty
                    nbnd[1] = (nel) // 2 + nempty

                occ = np.zeros((nspin, max(nbnd)), dtype=np.int32)
                if nspin == 1:
                    occ[nspin - 1, :nel // 2] = 2
                    occ[nspin - 1, nel // 2: nel // 2 + nel % 2] = 1
                else:
                    # spin up
                    occ[0, :(nel + 1) // 2] = 1
                    # spin down
                    occ[1, :nel // 2] = 1
            element.clear()

        context = etree.iterparse(file_name, huge_tree=True)

        index_mp = 0
        fileNameList_tot = [] 
        for isp in range(nspin):
            fileNameList_sp = [] 
            for iwf in range(nbnd[isp]):
                fileName = storeFolder + '/wfc_' + str(isp + 1) + '_' + str(iwf + 1).zfill(5) + '_r.npy'
                fileNameList_sp.append(fileName)
            fileNameList_tot.append(fileNameList_sp)
        fileNameList_tot = np.array(fileNameList_tot)
        if rank == 0:
            total_iter = np.sum(nbnd)
            pbar = tqdm(desc='store wfc', total=total_iter)
        for event, element in context:
            if element.tag == "grid_function":
                encoding = element.get("encoding")
                if index_mp % size == rank:
                    wfc_ = element.text
                    dtype = np.double
                    if encoding.strip() == "text":
                        wfc_flatten = np.fromstring(wfc_, dtype=dtype, sep=' ')
                        fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(iwfc + 1).zfill(5) + '_r'
                        np.save(fileName, wfc_flatten.reshape(fftw))
                    else:
                        wfc_byte = base64.decodebytes(bytes(wfc_, 'utf-8'))
                        wfc_flatten = np.frombuffer(wfc_byte, dtype=dtype)
                        fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(iwfc + 1).zfill(5) + '_r'
                        np.save(fileName, wfc_flatten.reshape(fftw))
                    if rank == 0:
                        value = size
                        if total_iter - index_mp < value:
                            value = total_iter - index_mp 
                        pbar.update(value)
                iwfc = (iwfc + 1) % nbnd[ispin]
                if iwfc == 0:
                    ispin += 1
                index_mp += 1
            element.clear()
        if rank == 0:
            pbar.close()

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
                    'occ': occ,
                    'fftw': fftw,
                    'npv': npv,
                    'wfc_file': fileNameList_tot}
        self.wfc_data = wfc_dict
        if rank == 0:
            with open(storeFolder + '/info.pickle', 'wb') as handle:
                pickle.dump(self.wfc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return wfc_dict

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
    # ----------------------------------Prepare------------------------------------

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xml", type=str,
            help="XML sample generated by qbox. Default: ../gs.gs.xml")
    parser.add_argument("-s", "--storeFolder", type=str,
            help="store wfc in Folder. Default: ./wfc/")
    args = parser.parse_args()

    # default values
    if not args.xml:
        args.xml = "../gs.gs.xml"
    if not args.storeFolder:
        args.storeFolder = "./wfc/"
    if rank == 0:
        print(f"configure:\
                \n {''.join(['-'] * 41)}\
                \n{'Qbox sample file':^20}:{args.xml:^20}\
                \n{'wfc storeFolder':^20}:{args.storeFolder:^20}\
                \n {''.join(['-'] * 41)}\n\
                ")
    # ---------------------------------Parse Qbox xml------------------------------------
    # test
    st = time.time()

    qbox = QBOXRead(comm=comm)
    qbox.parse_wfc(args.xml)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    comm.Barrier()
    if rank == 0:
        print('Execution time:', elapsed_time, 'seconds')
