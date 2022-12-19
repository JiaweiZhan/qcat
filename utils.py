import datetime
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def time_now():
    now = datetime.datetime.now()
    print("=====================================================")
    print("*              QE LDOS from wfc.dat                  ")
    print("*   Developed by Jiawei Zhan <jiaweiz@uchicago.edu>  ")
    print("*  Supported by Galli Group @ University of Chicago  ")
    print("* -- ")
    print("* Date: %s"%now)
    print("=====================================================")
    return

def writeLocalBandEdge(lcbm, lvbm, fileName='ldos.txt'):
    with open(fileName, 'w') as file_object:
        file_object.writelines("LVBM:\n")
        for i in range(len(lvbm)):
            file_object.write(f'{lvbm[i]:12.5f}')
            if i % 5 == 4:
                file_object.write('\n')
        file_object.write('\n\n')
        file_object.writelines("LCBM:\n")
        for i in range(len(lcbm)):
            file_object.write(f'{lcbm[i]:12.5f}')
            if i % 5 == 4:
                file_object.write('\n')

def drawLocalBandEdge(lcbm, lvbm, z_length, kernel_size=15, picName='ldos.pdf'):
    if kernel_size % 2 == 0:
        kernel_size += 1
    fig, ax = plt.subplots(figsize=(10, 5))
    x_axis = np.linspace(0, z_length, len(lcbm))
    # smooth
    # lvbm_smooth = savgol_filter(lvbm, 11, 3)
    # lcbm_smooth = savgol_filter(lcbm, 11, 3)
    lvbm = list(lvbm)
    lcbm = list(lcbm)

    kernel = np.ones(kernel_size) / kernel_size
    lvbm_smooth = lvbm[-(kernel_size - 1)// 2:]
    lvbm_smooth.extend(lvbm)
    lvbm_smooth.extend(lvbm[:(kernel_size - 1) // 2 + 1])
    lvbm_smooth = np.convolve(lvbm_smooth, kernel, mode='same')[(kernel_size - 1) // 2 + 1 : -(kernel_size - 1) // 2]

    lcbm_smooth = lcbm[-(kernel_size - 1)// 2:]
    lcbm_smooth.extend(lcbm)
    lcbm_smooth.extend(lcbm[:(kernel_size - 1) // 2 + 1])
    lcbm_smooth = np.convolve(lcbm_smooth, kernel, mode='same')[(kernel_size - 1) // 2 + 1 : -(kernel_size - 1) // 2]

    plt.plot(x_axis, lcbm_smooth, 'c')
    plt.plot(x_axis, lvbm_smooth, 'c')
    plt.xlabel('z axis / bohr', fontsize=15)
    plt.ylabel('Energy Level / eV', fontsize=15)
    plt.xlim([0, z_length])
    plt.savefig(picName, dpi=1000, bbox_inches='tight')
