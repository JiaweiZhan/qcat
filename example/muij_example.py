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
    parser.add_argument("-r", "--rhoFile", type=str,
            help="file for rho. Default: ../rho")
    args = parser.parse_args()

    if not args.saveFileFolder:
        args.saveFileFolder = "../" 
    if not args.rhoFile:
        args.rhoFile = "../rho" 

    conf_tab = {"saveFileFolder": args.saveFileFolder,
                "rhoFile": args.rhoFile,
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

    rho, mu = utils.read_rho(args.rhoFile)

    if rank == 0:
        # output mu_intact.dat
        with open("./intactMu.dat", "wb") as file_obj:
            file_obj.write(bytes(fftw))
            mu.tofile(file_obj)
    comm.Barrier()

    mu = zoom(mu, fftw * 1.0 / npv, mode='wrap')

    for ispin in range(nspin):
        mu_ij = np.zeros((nbnd[ispin], nbnd[ispin]))
        if rank == 0:
            total_iter = nbnd[ispin]
            pbar = tqdm(desc=f'compute mu_ij. for spin:{ispin + 1:^3}/{nspin:^3}', total=total_iter)
        for ibnd_i in range(nbnd[ispin]): 
            fileName = fileNameList[ispin, ibnd_i]
            wfc_i = np.load(fileName)
            for ibnd_j in range(ibnd_i, nbnd[ispin]):
                if ibnd_j % size == rank:
                    fileName = fileNameList[ispin, ibnd_j]
                    wfc_j = np.load(fileName)
                    wfc_ij = wfc_i * wfc_j

                    mu_ij[ibnd_i, ibnd_j] = np.sum(mu * np.absolute(wfc_ij) ** 2) / np.sum(np.absolute(wfc_ij) ** 2)

            if rank == 0:
                value = 1
                pbar.update(value)
        if rank == 0:
            pbar.close()
        mu_ij_global = np.zeros_like(mu_ij)
        comm.Allreduce(mu_ij, mu_ij_global, op=MPI.SUM)
        diag = np.diagonal(mu_ij_global).copy()
        mu_ij_global += mu_ij_global.T
        mu_ij_global[np.arange(nbnd[ispin]), np.arange(nbnd[ispin])] = diag
        if rank == 0:
            sys.stdout.write(f"spin: {ispin} / {nspin}\n")
            for ibnd_i in range(nbnd[ispin]): 
                sys.stdout.write(f"band ind:{str(ibnd_i + 1):^10}\n")
                for ibnd_j in range(nbnd[ispin]):
                    strPre = f"mu_{str(ibnd_i + 1)}_{str(ibnd_j + 1)}"
                    sys.stdout.write(f"{strPre:^15}: {mu_ij_global[ibnd_i, ibnd_j]:5.2f}")
                    if ibnd_j % 5 == 4:
                        sys.stdout.write("\n")
                sys.stdout.write("\n")

    comm.Barrier()
    if rank == 0:
        shutil.rmtree(storeFolder)

