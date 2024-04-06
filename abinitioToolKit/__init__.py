from abinitioToolKit.io_kernel import *
from abinitioToolKit.pp import *
from abinitioToolKit.utils import *
USE_GPU = True
try:
    import cupy
    try:
        USE_GPU = cupy.cuda.is_available()
    except:
        USE_GPU = False
except ImportError:
    USE_GPU = False

__version__ = "1.0.0"

def header():
    """Prints welcome header."""
    import datetime

    print("abinitioToolKit version : ", __version__)
    print("Use GPU                 : ", USE_GPU)
    print("Today                   : ", datetime.datetime.today())


header()
