"""
this file contains the class for compute LF that can be used in LRS-DDH
"""
import numpy as np
import argparse
import os, time
from qcat.io_kernel import qbox_io
from qcat.utils import utils
import shutil
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from mpi4py import MPI
import pickle

class LF:
    # thread-related global value
    xml_data = None

    def __init__(self):
        pass

    def readWFC(self, wfc_folder='./wfc/'):
        with open(wfc_folder + '/info.pickle', 'rb') as handle:
            self.xml_data = pickle.load(handle)

    def info(self):
        return self.xml_data

    def computeLF(self, epsilonPre, mus, mu_map, sigma=40, wfc_folder='./wfc/', comm=None):
        cell = self.xml_data['cell']
        nspin = self.xml_data['nspin']
        fftw = self.xml_data['fftw']
        nbnd = self.xml_data['nbnd']
        occ = self.xml_data['occ']
        nks = self.xml_data['nks']
        fileNameList = self.xml_data['wfc_file']
        storeFolder = wfc_folder

        rank = 0
        size = 1
        if comm:
            rank = comm.Get_rank()
            size = comm.Get_size()

        v_g = utils.vint(fftw, cell)

        # epsilon
        epsilon = zoom(epsilonPre, fftw * 1.0 / epsilonPre.shape, mode='wrap')
        epsilon = np.where(epsilon >= 1, epsilon, 1)

        # prepare mu and floorFunc
        v_g_mu = []
        floorFunc = []
        mus = np.array(mus)
        for index_mu, mu in enumerate(mus):
            v_g_mu.append(utils.vint_erfc(fftw, cell, mu))
            floorFunc_mu = np.zeros(fftw, np.int32)
            for i in range(fftw[0]):
                floorIndexI = int(i * 1.0 / fftw[0] * mu_map.shape[0])
                for j in range(fftw[1]):
                    floorIndexJ = int(j * 1.0 / fftw[1] * mu_map.shape[1])
                    for k in range(fftw[2]):
                        floorIndexK = int(k * 1.0 / fftw[2] * mu_map.shape[2])
                        floorFunc_mu[i, j, k] = 1 if mu_map[floorIndexI, floorIndexJ, floorIndexK] == index_mu else 0
            floorFunc.append(floorFunc_mu)

        lfFolder = './lf/'
        if rank == 0:
            isExist = os.path.exists(lfFolder)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(lfFolder)
        if comm:
            comm.Barrier()
        for ispin in range(nspin):
            # TODO: Possible multithread here
            for ibnd_i in range(nbnd[ispin]): 
                if ibnd_i % size == rank:
                    upper, lower = np.zeros(fftw), np.zeros(fftw)
                    for iks in range(nks):
                        fileName = fileNameList[ispin, iks, ibnd_i]
                        wfc_i = np.load(fileName)
                        for ibnd_j in range(nbnd[ispin]): 
                            fileName = fileNameList[ispin, iks, ibnd_j]
                            wfc_j = np.load(fileName)
                            wfc_ij = wfc_i * wfc_j
                            wfc_ij_g = np.fft.fftn(wfc_ij, norm='forward') 
                            wfc_ijeps_g = np.fft.fftn(wfc_ij * np.sqrt(1.0 / epsilon), norm='forward') 
                            wfc_ij1meps_g = np.fft.fftn(wfc_ij * np.sqrt(1.0 - 1.0 / epsilon), norm='forward') 
                            lower += wfc_j * np.real(np.fft.ifftn(v_g * wfc_ij_g, norm='forward')) * occ[ispin, ibnd_j]
                            sumPar = np.zeros(fftw)
                            for index_mu in range(mus.shape[0]):
                                sumPar += np.real(np.fft.ifftn(v_g_mu[index_mu] * wfc_ij1meps_g, norm='forward')) * floorFunc[index_mu]
                            upper += wfc_j * np.sqrt(1.0 - 1.0 / epsilon) * sumPar * occ[ispin, ibnd_j]
                            upper += wfc_j * np.sqrt(1.0 / epsilon) * np.real(np.fft.ifftn(v_g * wfc_ijeps_g, norm='forward')) * occ[ispin, ibnd_j]
                    lf = np.divide(upper, lower, out=np.zeros_like(upper), where=lower!=0)

                    lf_lower = np.percentile(lf, 1)
                    lf_higher = np.percentile(lf, 99)
                    lf = np.where(lf >= lf_lower, lf, lf_lower)
                    lf = np.where(lf <= lf_higher, lf,  lf_higher)
                    lf = gaussian_filter(lf, sigma=sigma, mode='wrap')
                    lf = zoom(lf, npv / fftw, mode='wrap', prefilter=False)

                    fileName = lfFolder + '/lf_' + str(ispin + 1) + '_' + str(ibnd_i + 1).zfill(5) + '.dat'
                    lfFile = open(fileName, 'wb')
                    lfFile.write(bytes(np.array(list(lf.shape), dtype=np.int32)))
                    for k in range(lf.shape[-1]):
                        lfFile.write(bytes(lf[:, :, k].flatten()))
                    lfFile.close()

        if comm:
            comm.Barrier()
        if rank == 0:
            shutil.rmtree(storeFolder)

if __name__=="__main__":
    # get the start time
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----------------------------------Prepare------------------------------------
    if rank == 0:
        utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--saveFileFolder", type=str,
            help="folder that store XML and qbox.out. Default: ../")
    parser.add_argument("-l", "--LocaFunctionFolder", type=str,
            help="The Folder that store local function, such Local dielectric. Default: ../")
    parser.add_argument("-s", "--Sigma", type=int,
            help="Sigma to guassian filter the local function. Default: 40")
    args = parser.parse_args()
    # default values
    if not args.saveFileFolder:
        args.saveFileFolder = "../"
    if not args.LocaFunctionFolder:
        args.LocaFunctionFolder = "../"
    if not args.Sigma:
        args.Sigma = 40 

    if rank == 0:
        print(f"configure:\
                \n {''.join(['-'] * 41)}\
                \n{'Qbox sample file':^20}:{args.xml:^20}\
                \n{'local function folder':^20}:{args.LocaFunctionFolder:^20}\
                \n{'Sigma':^20}:{args.Sigma:^20}\
                \n {''.join(['-'] * 41)}\n\
                ")

    st = time.time()

    # ------------------------------------------- read and store wfc --------------------------------------------
    
    qbox = qbox_io.QBOXRead(comm=comm, outFolder=args.saveFileFolder)
    storeFolder = './wfc/'

    comm.Barrier()
    isExist = os.path.exists(storeFolder)
    if not isExist:
        if rank == 0:
            print(f"store wfc from {storeFolder}")
        qbox.read(storeFolder=storeFolder, store_wfc=True)
    else:
        if rank == 0:
            print(f"read stored wfc from {storeFolder}")
     
    # ---------------------------------Compute LF------------------------------------

    lf = LF()
    lf.readWFC()
    wfc_data = lf.info()
    npv = wfc_data['npv']

    # read epsilon
    epsilon_file = args.LocaFunctionFolder + '/alpha.txt'
    spread_domain = args.LocaFunctionFolder + '/spread_domain.txt'
    domain_map = args.LocaFunctionFolder + '/domain_map.txt'
    epsilon = np.zeros(npv)
    with open(epsilon_file, 'r') as file_obj:
        line = file_obj.readline()
        index_line = 0
        while line:
            epsilon[:, :, index_line] = np.fromstring(line, sep=' ').reshape([npv[0], npv[1]])
            line = file_obj.readline()
            index_line += 1
    assert(np.all(epsilon >= 1))
    mus = None
    with open(spread_domain, 'r') as file_obj:
        file_obj.readline()
        line = file_obj.readline()
        mus = np.fromstring(line, sep=' ')
    mu_map = None
    with open(domain_map, 'r') as file_obj:
        grid = int(file_obj.readline())
        line = file_obj.readline()
        mu_map = np.fromstring(line, sep=' ').reshape([grid, grid, grid])

    lf.computeLF(epsilon, mus, mu_map, sigma=args.Sigma, comm=comm)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    if rank == 0:
        print('Execution time:', elapsed_time, 'seconds')
