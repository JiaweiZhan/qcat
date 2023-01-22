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
import shutil, os

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
    parser.add_argument("-a", "--alphaFile", type=str,
            help="Local Dielectric Function File. Default: ../alpha.txt")
    args = parser.parse_args()

    if not args.saveFileFolder:
        args.saveFileFolder = "../" 
    if not args.alphaFile:
        args.alphaFile = "../alpha.txt" 

    conf_tab = {"saveFileFolder": args.saveFileFolder,
                "alphaFile": args.alphaFile,
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
    fileNameList = info_data['wfc_file']

    # TODO: 1. Qbox has no k point; 2. qe has no different nbnd
    if len(fileNameList.shape) == 3:
        fileNameList = fileNameList[:, 0, :]

    alphaFile = args.alphaFile 
    alpha = utils.read_alpha(alphaFile=alphaFile, npv=npv)
    alpha = zoom(alpha, fftw * 1.0 / alpha.shape, mode='wrap')
    alpha = np.where(alpha >= 1, alpha, 1)

    for ispin in range(nspin):
        die_feel = np.zeros(nbnd[ispin])
        if rank == 0:
            total_iter = nbnd[ispin]
            pbar = tqdm(desc=f'compute dielectric. for spin:{ispin + 1:^3}/{nspin:^3}', total=total_iter)
        for ibnd_i in range(nbnd[ispin]): 
            if ibnd_i % size == rank:
                fileName = fileNameList[ispin, ibnd_i]
                wfc_i = np.load(fileName)
                wfc_i = np.absolute(wfc_i) ** 2
                die_feel[ibnd_i] = np.sum(1.0 / alpha * wfc_i) / np.sum(wfc_i)

                if rank == 0:
                    value = size
                    if nbnd[ispin] - ibnd_i < value:
                        value = nbnd[ispin] - ibnd_i
                    pbar.update(value)
        if rank == 0:
            pbar.close()
        die_feel_global = np.zeros_like(die_feel)
        comm.Allreduce(sendbuf=die_feel, recvbuf=die_feel_global, op=MPI.SUM)
        if rank == 0:
            for ibnd_i in range(nbnd[ispin]): 
                print(f"spin: {ispin} / {nspin}|| band ind:{str(ibnd_i + 1):^10}: {die_feel_global[ibnd_i]:10.5f}")
    comm.Barrier()
    if rank == 0:
        shutil.rmtree(storeFolder)

