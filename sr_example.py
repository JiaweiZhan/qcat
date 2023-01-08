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
    parser.add_argument("-s", "--saveFileFolder", type=str,
            help="output From QC software: *.save for qe QE .xml for Qbox. Default: ./scf.save")
    args = parser.parse_args()

    if not args.abinitio:
        args.abinitio = "qbox"
    if not args.saveFileFolder:
        args.saveFileFolder = "./scf.save" 

    if rank == 0:
        print(f"configure:\
                \n {''.join(['-'] * 41)}\
                \n{'software':^20}:{args.abinitio:^20}\
                \n{'saveFileFolder':^20}:{args.saveFileFolder:^20}\
                \n {''.join(['-'] * 41)}\n\
                ")

    abinitioRead = None
    if args.abinitio.lower() == "qbox":
        abinitioRead = qbox_io.QBOXRead(comm)
    elif args.abinitio.lower() == "qe":
        abinitioRead = qe_io.QERead(comm)
    utils.local_contribution(abinitioRead, saveFileFolder=args.saveFileFolder, comm=comm)
