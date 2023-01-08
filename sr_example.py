from abinitioToolKit import qbox_io
from abinitioToolKit import utils
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

if __name__ == "__main__":
    qbox = qbox_io.QBOXRead(comm)
    utils.local_contribution(qbox, info_name=None, wfc_name='../gs.gs.xml', comm=comm)
