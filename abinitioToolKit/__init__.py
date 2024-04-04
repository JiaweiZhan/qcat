from abinitioToolKit.io_kernel import *
from abinitioToolKit.pp import *
from abinitioToolKit.utils import *
USE_CUPY = True
try:
    import cupy
except ImportError:
    USE_CUPY = False

__version__ = "1.0.0"

def header():
    """Prints welcome header."""
    import datetime

    print("abinitioToolKit version : ", __version__)
    print("Use GPU                 : ", USE_CUPY)
    print("Today                   : ", datetime.datetime.today())


header()
