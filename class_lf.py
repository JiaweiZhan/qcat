"""
this file contains the class for compute LF that can be used in LRS-DDH
"""
import numpy as np
from tqdm import tqdm
import os, time, pathlib
import threading
import qbox_io
import utils
import shutil

class LF:
    # thread-related global value
    qbox_xml = None
    xml_data = None

    def __init__(self, qbox_xml):
        self.qbox_xml = qbox_xml

    def readWFC(self, wfc_folder='./wfc/'):
        qboxInput = qbox_io.QBOXRead()
        self.xml_data = qboxInput.parse_QBOX_XML(self.qbox_xml)
        qboxInput.storeWFC(storeFolder=wfc_folder)


    def info(self):
        return self.xml_data

    def computeLF(self, epsilonPre, mus, mu_map, wfc_folder='./wfc/'):
        cell = self.xml_data['cell']
        nspin = self.xml_data['nspin']
        fftw = self.xml_data['fftw']
        nbnd = self.xml_data['nbnd']
        occ = self.xml_data['occ']
        storeFolder = wfc_folder

        v_g = utils.vint(fftw, cell)

        # epsilon and mu
        # scale epsilon
        epsilon = np.zeros(fftw)
        for i in range(fftw[0]):
            floorIndexI = int(i * 1.0 / fftw[0] * mu_map.shape[0])
            for j in range(fftw[1]):
                floorIndexJ = int(j * 1.0 / fftw[1] * mu_map.shape[1])
                for k in range(fftw[2]):
                    floorIndexK = int(k * 1.0 / fftw[2] * mu_map.shape[2])
                    epsilon[i, j, k] = epsilonPre[floorIndexI, floorIndexJ, floorIndexK]

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
        isExist = os.path.exists(lfFolder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(lfFolder)
        for ispin in range(nspin):
            # TODO: Possible multithread here
            for ibnd_i in tqdm(range(nbnd), desc='compute lf'): 
                fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(ibnd_i + 1).zfill(5) + '_r' + '.npy'
                wfc_i = np.load(fileName)
                upper, lower = np.zeros(fftw), np.zeros(fftw)
                for ibnd_j in range(nbnd): 
                    fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(ibnd_j + 1).zfill(5) + '_r' + '.npy'
                    wfc_j = np.load(fileName)
                    wfc_ij = wfc_i * wfc_j
                    wfc_ij_g = np.fft.fftn(wfc_ij, norm='forward') 
                    lower += wfc_j * np.real(np.fft.ifftn(v_g * wfc_ij_g, norm='forward')) * occ[ispin, ibnd_j]
                    sumPar = np.zeros(fftw)
                    for index_mu in range(mus.shape[0]):
                        sumPar += np.real(np.fft.ifftn(v_g_mu[index_mu] * wfc_ij_g, norm='forward')) * floorFunc[index_mu]
                    upper += wfc_j * sumPar * occ[ispin, ibnd_j]
                lf = np.divide(upper, lower, out=np.zeros_like(upper), where=lower!=0)
                fileName = lfFolder + '/lf_' + str(ispin + 1) + '_' + str(ibnd_i + 1).zfill(5) + '.dat'
                lfFile = open(fileName, 'wb')
                lfFile.write(bytes(np.array(list(lf.shape), dtype=np.int32)))
                lfFile.write(bytes(lf.flatten()))
                lfFile.close()
        shutil.rmtree(storeFolder)

if __name__=="__main__":
    # get the start time
    st = time.time()

    lf = LF("../diamond.gs.xml")
    lf.readWFC()
    wfc_data = lf.info()
    npv = wfc_data['npv']

    # read epsilon
    epsilon_file = '../alpha.txt'
    spread_domain = '../spread_domain.txt'
    domain_map = '../domain_map.txt'
    epsilon = np.zeros(npv)
    with open(epsilon_file, 'r') as file_obj:
        line = file_obj.readline()
        index_line = 0
        while line:
            epsilon[:, :, index_line] = np.fromstring(line, sep=' ').reshape([npv[0], npv[1]])
            line = file_obj.readline()
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

    lf.computeLF(epsilon, mus, mu_map)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
