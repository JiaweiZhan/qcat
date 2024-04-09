# only support qbox yet
from abinitioToolKit import qbox_io
from abinitioToolKit import utils
from mpi4py import MPI
import argparse
from functools import partial
import signal
import pickle
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
import shutil, os, sys
from collections import OrderedDict

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    signal.signal(signal.SIGINT, partial(utils.handler, comm))

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--saveFileFolder", type=str,
            help="folder that store XML sample and qbox.out. Default: ../")
    parser.add_argument("-d", "--spread_domain", type=str,
            help="spread_domain file. Default: ../spread_domain.txt")
    parser.add_argument("-m", "--domain_map", type=str,
            help="domain_map file. Default: ../domain_map.txt")
    args = parser.parse_args()

    if not args.saveFileFolder:
        args.saveFileFolder = "../" 
    if not args.spread_domain:
        args.spread_domain = "../spread_domain.txt" 
    if not args.domain_map:
        args.domain_map = "../domain_map.txt" 

    conf_tab = {"saveFileFolder": args.saveFileFolder,
                "spread_domain": args.spread_domain,
                "domain_map": args.domain_map,
                "MPI size": comm.Get_size()}
    utils.print_conf(conf_tab)

    # ------------------------------------------- read and store wfc --------------------------------------------
    
    qbox = qbox_io.QBOXRead(comm=comm)
    storeFolder = './wfc/'

    comm.Barrier()
    isExist = os.path.exists(storeFolder)
    if not isExist:
        if rank == 0:
            print(f"store wfc from {storeFolder}")
        qbox.read(args.saveFileFolder, storeFolder=storeFolder)
    else:
        if rank == 0:
            print(f"read stored wfc from {storeFolder}")
     
    # --------------------------------------- compute alpha felt by states----------------------------------------
    
    # comm.Barrier()
    with open(storeFolder + '/info.pickle', 'rb') as handle:
        info_data = pickle.load(handle)

    npv = info_data['npv']
    fftw = info_data['fftw']
    nbnd = info_data['nbnd']
    nspin = info_data['nspin']
    occ = info_data['occ']
    cell = info_data['cell']
    fileNameList = info_data['wfc_file']

    # TODO: 1. Qbox has no k point; 2. qe has no different nbnd
    if len(fileNameList.shape) == 3:
        fileNameList = fileNameList[:, 0, :]
        occ = occ[:, 0, :]

    mus, mu_map = utils.read_mu(args.spread_domain, args.domain_map)
    mu_mapGeneral = np.zeros(fftw, dtype=np.int32)
    for i in range(fftw[0]):
        i_grid = int(i / fftw[0] * mu_map.shape[0])
        for j in range(fftw[1]):
            j_grid = int(j / fftw[1] * mu_map.shape[1])
            for k in range(fftw[2]):
                k_grid = int(k / fftw[2] * mu_map.shape[2])
                mu_mapGeneral[i, j, k] = mu_map[i_grid, j_grid, k_grid]
    floorMu = []
    for i in range(len(mus)):
        floorMu_ = np.where(mu_mapGeneral == i, 1, 0)
        floorMu.append(floorMu_)

    for ispin in range(nspin):
        if rank == 0:
            total_iter = nbnd[ispin]
            pbar = tqdm(desc=f'compute muDomain_ij. for spin:{ispin + 1:^3}/{nspin:^3}', total=total_iter)
        muDomainDict = {}
        for ibnd_i in range(nbnd[ispin]): 
            fileName = fileNameList[ispin, ibnd_i]
            wfc_i = np.load(fileName)
            for ibnd_j in range(ibnd_i, nbnd[ispin]):
                if ibnd_j % size == rank:
                    fileName = fileNameList[ispin, ibnd_j]
                    wfc_j = np.load(fileName)
                    wfc_ij = wfc_i * wfc_j

                    mu_ratio = []
                    for idx, mu in enumerate(mus):
                        ratio = np.sum(floorMu[idx] * np.absolute(wfc_ij) ** 2) / np.sum(np.absolute(wfc_ij) ** 2)
                        mu_ratio.append([mu, ratio])
                    mu_ratio = sorted(mu_ratio, key=lambda x: x[1], reverse=True)
                    threshold = 0.9
                    sum = 0
                    id = 0
                    for ratio in mu_ratio:
                        id += 1
                        sum += ratio[-1]
                        if sum >= threshold:
                            break

                    muDomainDict[f"{ibnd_i}/{ibnd_j}"] = mu_ratio[:id]

            if rank == 0:
                value = 1
                pbar.update(value)
        if rank == 0:
            pbar.close()
        comm.Barrier()
        muDomainDict = comm.gather(muDomainDict, root=0)

        if rank == 0:
            # output mu_intact.dat
            with open("./ExxMuDomain" + str(ispin + 1) + ".dat", "w") as file_obj:
                file_obj.write(str(nbnd[ispin]) + '\n')
                for dict_ in muDomainDict:
                    for key, value in dict_.items():
                        file_obj.write(key + '\t:')
                        for value_ in list(value):
                            file_obj.write(",".join(f"{num:7.3f}" for num in value_))
                            file_obj.write("\t")
                        file_obj.write("\n")

    comm.Barrier()
    if rank == 0:
        shutil.rmtree(storeFolder)

