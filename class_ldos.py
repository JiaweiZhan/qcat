import numpy as np
from tqdm import tqdm
import os, time, pathlib
import threading
import qe_io
import shutil
from mpi4py import MPI

class LDOS:

    def __init__(self, delta=0.001, saveFolder='./scf.save', comm=None):
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

    def computeLDOS(self, storeFolder='./wfc'):

        self.storeFolder = storeFolder 
        rank, size = 0, 1
        if not self.comm is None:
            rank = self.comm.Get_rank()
            size = self.comm.Get_size()
        qe = qe_io.QERead(self.comm)
        xml_data = qe.parse_QE_XML(self.saveFolder + '/data-file-schema.xml')
        numOcc = xml_data['occ']
        kWeights = xml_data['kweights']
        nbnd = xml_data['nbnd']
        eigens = xml_data['eigen']
        nspin = numOcc.shape[0]
        nks = numOcc.shape[1]
        nbnd = numOcc.shape[2]
        fft_grid = xml_data['fftw']
        fft_grid = np.array(fft_grid) // 2 + 1

        if rank == 0:
            if np.all(numOcc > 0):
                print("no conduction band")
                isExist = os.path.exists(self.storeFolder)
                if isExist:
                    shutil.rmtree(self.storeFolder)
                self.comm.Abort()
        self.comm.Barrier()

        ksStateZAve_loc = np.zeros((nspin, nks, nbnd, fft_grid[2]), dtype=np.double)
        wfc_files = [self.saveFolder + '/' + file for file in os.listdir(self.saveFolder) if 'wfc' in file]

        for index in range(nks * nspin):
            wfc_data = qe.parse_QE_wfc(wfc_files[index], storeFolder=storeFolder)
            self.comm.Barrier()
            ispin = wfc_data['ispin']
            ik = wfc_data['ik']

            wfcStored = [name for name in os.listdir(storeFolder) if "wfc" in name]
            for index, fileName in enumerate(wfcStored):
                if index % size == rank:
                    wfcName = storeFolder + '/' + fileName
                    ibnd = int(fileName.split('.')[0].split('_')[-2])
                    evc_r = np.load(wfcName)
                    ksStateZAve_loc[ispin - 1, ik - 1, ibnd - 1, :] = np.sum(np.absolute(evc_r) ** 2, axis=(0, 1,))

            self.comm.Barrier()
            if rank == 0:
                shutil.rmtree(storeFolder)
            self.comm.Barrier()
        ksStateZAve = np.zeros_like(ksStateZAve_loc)
        self.comm.Allreduce(ksStateZAve_loc, ksStateZAve)

        lcbm_loc = np.zeros(fft_grid[2])
        lvbm_loc = np.zeros(fft_grid[2])

        for z in range(fft_grid[2]):
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

        self.comm.Allreduce(lcbm_loc, self.lcbm)
        self.comm.Allreduce(lvbm_loc, self.lvbm)

    def localBandEdge(self):
        return self.lcbm, self.lvbm

if __name__=="__main__":
    # get the start time
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    st = time.time()

    qe = qe_io.QERead(comm)
    qe.parse_QE_XML('../bn.save/data-file-schema.xml')
    qe.parse_QE_wfc('../bn.save/wfc1.dat')

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    comm.Barrier()
    if rank == 0:
        print('Execution time:', elapsed_time, 'seconds')
