import numpy as np
from tqdm import tqdm
import os, time, pathlib
import threading
from qcat.io_kernel import qe_io
import shutil
from mpi4py import MPI
import pickle

class LDOS:

    def __init__(self, read_obj, delta=0.001, saveFolder='./scf.save', comm=None):
        """
        Init LDOS class:
            param:
                delta: float;
                numThread: int;
                saveFolder: str
                backp: bool
        """
        self.delta = delta
        self.lvbm = None 
        self.lcbm = None 
        self.saveFolder = saveFolder
        self.storeFolder = './wfc/'
        self.comm = comm
        self.read_obj = read_obj

    def computeLDOS(self, storeFolder='./wfc', axis='z'):
        assert axis in ['x', 'y', 'z']

        self.storeFolder = storeFolder
        rank, size = 0, 1
        if self.comm:
            rank = self.comm.Get_rank()
            size = self.comm.Get_size()
        self.read_obj.read(storeFolder=self.storeFolder, real_space=True)

        with open(storeFolder + '/info.pickle', 'rb') as handle:
            xml_data = pickle.load(handle)

        numOcc = xml_data['occ']
        kWeights = xml_data['kweights']
        eigens = xml_data['eigen']
        nspin = numOcc.shape[0]
        nks = numOcc.shape[1]
        nbnd = numOcc.shape[2]
        fft_grid = xml_data['fftw']

        ldos_grid = 0
        axis_ave = (0, 1,)
        if axis == 'z':
            ldos_grid = fft_grid[2]
            axis_ave = (0, 1,)
        elif axis == 'y':
            ldos_grid = fft_grid[1]
            axis_ave = (0, 2,)
        elif axis == 'x':
            ldos_grid = fft_grid[0]
            axis_ave = (1, 2,)
        else:
            raise NotImplementedError

        if rank == 0:
            if np.all(numOcc > 0):
                print("no conduction band")
                isExist = os.path.exists(self.storeFolder)
                if isExist:
                    shutil.rmtree(self.storeFolder)
                if self.comm:
                    self.comm.Abort()
                else:
                    assert False
        if self.comm:
            self.comm.Barrier()

        ksStateZAve_loc = np.zeros((nspin, nks, nbnd, ldos_grid), dtype=np.double)

        wfcStored = [name for name in os.listdir(storeFolder) if "wfc" in name]

        for index, fileName in enumerate(wfcStored):
            if index % size == rank:
                wfcName = storeFolder + '/' + fileName
                ibnd = int(fileName.split('.')[0].split('_')[-2])
                ik = int(fileName.split('.')[0].split('_')[-3])
                ispin = int(fileName.split('.')[0].split('_')[-4])
                evc_r = np.load(wfcName)
                ksStateZAve_loc[ispin - 1, ik - 1, ibnd - 1, :] = np.sum(np.absolute(evc_r) ** 2, axis=axis_ave)

        ksStateZAve = np.zeros_like(ksStateZAve_loc)
        if self.comm:
            self.comm.Allreduce(ksStateZAve_loc, ksStateZAve)
        else:
            ksStateZAve = ksStateZAve_loc

        lcbm_loc = np.zeros(ldos_grid)
        lvbm_loc = np.zeros(ldos_grid)

        for z in range(ldos_grid):
            if z % size == rank:
                # ksStateZAve: [ispin, ik, ibnd, z]
                preFactor = ksStateZAve[:, :, :, z]
                sumVBTot = np.sum(preFactor * numOcc * kWeights[np.newaxis, :, np.newaxis])

                KSEnergyTot = []
                KSFactorTot = []
                for i in range(nspin):
                    for j in range(nks):
                        KSEnergyTot.extend(eigens[i, j])
                        KSFactorTot.extend(preFactor[i, j] * kWeights[j])

                zipEneFac = zip(KSEnergyTot, KSFactorTot)
                eneSort, facSort = list(zip(*sorted(zipEneFac, key=lambda x: x[0])))

                min_arg = int(np.sum(numOcc)) - 1
                max_arg = int(np.sum(numOcc))

                sumLeft = 0
                while min_arg >= 1:
                    sumLeft += facSort[min_arg]
                    if sumLeft >= sumVBTot * self.delta:
                        break
                    else:
                        min_arg -= 1
                if min_arg != int(np.sum(numOcc)) -1:
                    min_arg += 1

                sumRight = 0
                while max_arg <= len(eneSort) - 2:
                    sumRight += facSort[max_arg]
                    if sumRight >= sumVBTot * self.delta:
                        break
                    else:
                        max_arg += 1
                if max_arg != int(np.sum(numOcc)):
                    max_arg -= 1
                lvbm_loc[z] = eneSort[min_arg]
                lcbm_loc[z] = eneSort[max_arg]
        self.lcbm = np.zeros_like(lcbm_loc)
        self.lvbm = np.zeros_like(lvbm_loc)

        if self.comm:
            self.comm.Allreduce(lcbm_loc, self.lcbm)
            self.comm.Allreduce(lvbm_loc, self.lvbm)
        else:
            self.lcbm = lcbm_loc
            self.lvbm = lvbm_loc
        self.read_obj.clean_wfc(storeFolder=self.storeFolder)

    def localBandEdge(self):
        return self.lcbm, self.lvbm

if __name__=="__main__":
    # get the start time
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    st = time.time()

    qe = qe_io.QERead(comm = comm, outFolder="../../bn.save")
    qe.parse_info()
    qe.parse_wfc()

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    comm.Barrier()
    if rank == 0:
        print('Execution time:', elapsed_time, 'seconds')
