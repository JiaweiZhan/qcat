USE_GPU = True
try:
    import cupy
    try:
        USE_GPU = cupy.cuda.is_available()
    except:
        USE_GPU = False
except ImportError:
    USE_GPU = False

import torch
import os

__version__ = "1.6.0"

def set_nthread():
    if USE_GPU:
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(os.cpu_count() - 1)

def header():
    """Prints welcome header."""
    import datetime

    print("qcat version            : ", __version__)
    print("Use GPU                 : ", USE_GPU)
    print("Today                   : ", datetime.datetime.today())


header()
set_nthread()
