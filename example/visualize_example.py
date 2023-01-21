# only support qbox yet
from mpi4py import MPI
import argparse
from functools import partial
import signal
import pickle
from tqdm import tqdm
import numpy as np
import shutil, os, sys

from abinitioToolKit import qbox_io
from abinitioToolKit import utils

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
    parser.add_argument("-b", "--bandIndex", type=int,
            help="band index. Default: 1")
    args = parser.parse_args()

    if not args.saveFileFolder:
        args.saveFileFolder = "./gs.gs.xml" 
    if not args.bandIndex:
        args.bandIndex = 1 

    conf_tab = {"saveFileFolder": args.saveFileFolder,
                "band index": args.bandIndex,
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
        fileName = fileNameList[ispin, args.bandIndex - 1]
        wfc = np.load(fileName)
        fileName = str(ispin + 1) + "_" + str(args.bandIndex).zfill(5) + ".dat"
        utils.visualize_func(np.absolute(wfc) ** 2, zoom_factor=0.5, fileName=fileName)
        if rank == 0:
            print(f"output func in {fileName}")
    comm.Barrier()
    if rank == 0:
        shutil.rmtree(storeFolder)
