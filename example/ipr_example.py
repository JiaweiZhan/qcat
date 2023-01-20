# only support qbox yet
from mpi4py import MPI
import argparse
from functools import partial
import signal
import pickle
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
import shutil, os, sys

# module_path = os.path.abspath('../')
# sys.path.append(module_path)
# from abinitioToolKit import qbox_io
# from abinitioToolKit import utils

from abinitioToolKit.abinitioToolKit import qbox_io

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    signal.signal(signal.SIGINT, partial(utils.handler, comm))

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--saveFileFolder", type=str,
            help="output From QC software: .xml for Qbox. Default: ./gs.gs.xml")
    args = parser.parse_args()

    if not args.saveFileFolder:
        args.saveFileFolder = "./gs.gs.xml" 

    conf_tab = {"saveFileFolder": args.saveFileFolder,
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
     
    # --------------------------------------- compute IPR by states----------------------------------------
    
    # comm.Barrier()
    with open(storeFolder + '/info.pickle', 'rb') as handle:
        info_data = pickle.load(handle)

    npv = info_data['npv']
    fftw = info_data['fftw']
    nbnd = info_data['nbnd']
    nspin = info_data['nspin']
    fileNameList = info_data['wfc_file']

    # TODO: 1. Qbox has no k point; 2. qe has no different nbnd
    try:
        nbnd[0]
    except TypeError:
        nbnd = [nbnd] * nspin
    if len(fileNameList.shape) == 3:
        fileNameList = fileNameList[:, 0, :]

    for ispin in range(nspin):
        ipr_loc = np.zeros(nbnd[ispin])
        if rank == 0:
            total_iter = nbnd[ispin]
            pbar = tqdm(desc=f'compute IPR for spin:{ispin + 1:^3}/{nspin:^3}', total=total_iter)
        for ibnd_i in range(nbnd[ispin]): 
            if ibnd_i % size == rank:
                fileName = fileNameList[ispin, ibnd_i]
                wfc_i = np.load(fileName)
                ipr_loc[ibnd_i] = np.sum(np.absolute(wfc_i) ** 4) / (np.sum(np.absolute(wfc_i) ** 2) ** 2)

                if rank == 0:
                    value = size
                    if nbnd[ispin] - ibnd_i < value:
                        value = nbnd[ispin] - ibnd_i
                    pbar.update(value)
        if rank == 0:
            pbar.close()
        ipr = np.zeros_like(ipr_loc)
        comm.Allreduce(sendbuf=ipr_loc, recvbuf=ipr, op=MPI.SUM)
        if rank == 0:
            for ibnd_i in range(nbnd[ispin]): 
                print(f"spin: {ispin} / {nspin}|| band ind:{str(ibnd_i + 1):^10}: {ipr[ibnd_i]:10.5f}")
    comm.Barrier()
    if rank == 0:
        shutil.rmtree(storeFolder)

