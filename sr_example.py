from abinitioToolKit import qbox_io
from abinitioToolKit import qe_io
from abinitioToolKit import utils
from mpi4py import MPI
import argparse

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    rank = comm.Get_rank()
    if rank == 0:
        utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--abinitio", type=str,
            help="abinitio software: qe/qbox. Default: qbox")
    parser.add_argument("-i", "--info", type=str,
            help="file for basic info: data.xml from qe and none for qbox. Default: None")
    parser.add_argument("-w", "--wfc", type=str,
            help="file/folder for wfc --- qe: save folder; qbox: xml file. Default: ./gs.gs.xml")
    args = parser.parse_args()

    if not args.abinitio:
        args.abinitio = "qbox"
    if not args.info:
        args.info = "None" 
    if not args.wfc:
        args.wfc = "./gs.gs.xml" 

    if rank == 0:
        print(f"configure:\
                \n {''.join(['-'] * 41)}\
                \n{'software':^20}:{args.abinitio:^20}\
                \n{'info':^20}:{args.info:^20}\
                \n{'wfc':^20}:{args.wfc:^20}\
                \n {''.join(['-'] * 41)}\n\
                ")

    abinitioRead = None
    if args.abinitio.lower() == "qbox":
        abinitioRead = qbox_io.QBOXRead(comm)
    elif args.abinitio.lower() == "qe":
        abinitioRead = qe_io.QERead(comm)
    if args.info == "None":
        args.info = None
    utils.local_contribution(abinitioRead, info_name=args.info, wfc_name=args.wfc, comm=comm)
