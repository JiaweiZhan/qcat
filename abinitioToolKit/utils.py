import datetime
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from mpi4py import MPI
import pickle
from tqdm import tqdm
import shutil, os

def time_now():
    now = datetime.datetime.now()
    print("=====================================================")
    print("*               abinitioToolKit                      ")
    print("*   Developed by Jiawei Zhan <jiaweiz@uchicago.edu>  ")
    print("*  Supported by Galli Group @ University of Chicago  ")
    print("* -- ")
    print("* Date: %s"%now)
    print("=====================================================")
    return

def print_conf(conf_tab):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print("configure:")
        print(''.join(['-'] * 41))
        for key, value in conf_tab.items():
            print(f"{key:^20}: {str(value):^20}")
        print(''.join(['-'] * 41))

def handler(comm, signum, frame):
    # handler for handle ctrl-C
    rank = comm.Get_rank()
    if rank == 0:
        print("", end="\r", flush=True)
        print("clean store file", flush=True)
        isExist = os.path.exists('./wfc/')
        if isExist:
            print("cleand!", flush=True)
            shutil.rmtree('./wfc/')
    comm.Barrier()
    comm.Abort(1)

def read_alpha(alphaFile, npv):
    epsilon = np.zeros(npv)
    with open(alphaFile, 'r') as file_obj:
        line = file_obj.readline()
        index_line = 0
        while line:
            epsilon[:, :, index_line] = np.fromstring(line, sep=' ').reshape([npv[0], npv[1]])
            line = file_obj.readline()
            index_line += 1
    assert(np.all(epsilon >= 1))
    return epsilon

def read_mu(spread_domain, domain_map):
    mus = None
    with open(spread_domain, 'r') as file_obj:
        file_obj.readline()
        line = file_obj.readline()
        mus = np.fromstring(line, sep=' ')
    mu_map = None
    with open(domain_map, 'r') as file_obj:
        grid = int(file_obj.readline())
        line = file_obj.readline()
        mu_map = np.fromstring(line, sep=' ').reshape([grid, grid, grid])
    return mus, mu_map

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
    print(f"\n\nLocal band edge is printed in {fileName}")

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
    print(f"Local band edge is drew on {picName}")

def vint(fftw, cell):
    """
    fftw: [np0v, np1v, np2v]
    cell: [[],[],[]]
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    b = np.zeros((3, 3))
    volume = abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
    fac = 2.0 * np.pi
    b[0] = fac / volume * np.cross(cell[1], cell[2])
    b[1] = fac / volume * np.cross(cell[2], cell[0])
    b[2] = fac / volume * np.cross(cell[0], cell[1])
    G_vec_norm = np.zeros(fftw)
    fftFreq = [[int(num) for num in np.fft.fftfreq(fftw[i]) * fftw[i]] for i in range(3)]
    for i in range(fftw[0]):
        if i % size == rank:
            index_i = fftFreq[0][i]
            for j in range(fftw[1]):
                index_j = fftFreq[1][j]
                for k in range(fftw[2]):
                    index_k = fftFreq[2][k]
                    G_vec = index_i * b[0] + index_j * b[1] + index_k * b[2]
                    G_vec_norm[i, j, k] = np.linalg.norm(G_vec) ** 2
    G_vec_norm_global = np.zeros_like(G_vec_norm)
    comm.Allreduce(G_vec_norm, G_vec_norm_global, op=MPI.SUM)
    G_vec_norm = G_vec_norm_global
    index = np.argwhere(G_vec_norm == 0)
    G_vec_norm[index[:, 0], index[:, 1], index[:, 2]] = [1] * index.shape[0]
    G_vec_norm = 4.0 * np.pi / volume  / G_vec_norm
    L_help = ((2.0 * np.pi) ** 3.0 / volume * 3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
    divergence = 16.0 * np.pi ** 2 / ((2.0 * np.pi) ** 3.0) * L_help
    G_vec_norm[index[:, 0], index[:, 1], index[:, 2]] = [divergence] * index.shape[0]
    return G_vec_norm

def vint_erfc(fftw, cell, mu):
    """
    fftw: [np0v, np1v, np2v]
    cell: [[],[],[]]
    mu: 0.65
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    b = np.zeros((3, 3))
    volume = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
    fac = 2.0 * np.pi
    b[0] = fac / volume * np.cross(cell[1], cell[2])
    b[1] = fac / volume * np.cross(cell[2], cell[0])
    b[2] = fac / volume * np.cross(cell[0], cell[1])
    G_vec_norm = np.zeros(fftw)
    fftFreq = [[int(num) for num in np.fft.fftfreq(fftw[i]) * fftw[i]] for i in range(3)]
    for i in range(fftw[0]):
        if i % size == rank:
            index_i = fftFreq[0][i]
            for j in range(fftw[1]):
                index_j = fftFreq[1][j]
                for k in range(fftw[2]):
                    index_k = fftFreq[2][k]
                    G_vec = index_i * b[0] + index_j * b[1] + index_k * b[2]
                    G_vec_norm[i, j, k] = np.linalg.norm(G_vec) ** 2
    G_vec_norm_global = np.zeros_like(G_vec_norm)
    comm.Allreduce(G_vec_norm, G_vec_norm_global, op=MPI.SUM)
    G_vec_norm = G_vec_norm_global
    index = np.argwhere(G_vec_norm == 0)
    G_vec_norm[index[:, 0], index[:, 1], index[:, 2]] = [1] * index.shape[0]
    G_vec_norm = 4.0 * np.pi / volume  / G_vec_norm * (1.0 - np.exp(-G_vec_norm / 4.0 / mu ** 2))
    divergence = 1.0 / 4.0 / mu ** 2 / volume * 4 * np.pi
    G_vec_norm[index[:, 0], index[:, 1], index[:, 2]] = [divergence] * index.shape[0]
    return G_vec_norm

def factorizable(n):
    if n % 11 == 0:
        n /= 11
    if n % 7 == 0:
        n /= 7
    if n % 5 == 0:
        n /= 5
    if n % 3 == 0:
        n /= 3
    if n % 3 == 0:
        n /= 3
    while n % 2 == 0:
        n /= 2
    return n == 1

def local_contribution(read_obj, saveFileFolder, comm, storeFolder='./wfc/'):
    rank = comm.Get_rank()
    size = comm.Get_size()
    read_obj.parse_info(saveFileFolder)
    read_obj.parse_wfc(saveFileFolder, storeFolder=storeFolder)

    comm.Barrier()
    with open(storeFolder + '/info.pickle', 'rb') as handle:
        info_data = pickle.load(handle)

    if rank == 0:
        print("store wfc done!")

    nbnd = info_data['nbnd']
    nspin = info_data['nspin']
    cell = info_data['cell']
    fftw = info_data['fftw']
    occ = info_data['occ']
    fileNameList = info_data['wfc_file']

    # TODO: 1. Qbox has no k point; 2. qe has no different nbnd
    try:
        nbnd[0]
    except TypeError:
        nbnd = [nbnd] * nspin
    if len(occ.shape) == 3:
        occ = occ[:, 0, :]
        fileNameList = fileNameList[:, 0, :]

    v_g = vint(fftw, cell)
    v_g_mu = vint_erfc(fftw, cell, mu=0.71)

    lower_chunk, upper_chunk = 0, 0
    for ispin in range(nspin):
        if rank == 0:
            total_iter = nbnd[ispin]
            pbar = tqdm(desc=f'compute local contri. for spin:{ispin + 1:^3}/{nspin:^3}', total=total_iter)
        for ibnd_i in range(nbnd[ispin]): 
            if ibnd_i % size == rank:
                fileName = fileNameList[ispin, ibnd_i]
                wfc_i = np.load(fileName)
                for ibnd_j in range(ibnd_i, nbnd[ispin]): 
                    # fileName = storeFolder + '/wfc_' + str(ispin + 1) + '_' + str(ibnd_j + 1).zfill(5) + '_r' + '.npy'
                    fileName = fileNameList[ispin, ibnd_j]
                    wfc_j = np.load(fileName)
                    wfc_ij = wfc_i * wfc_j
                    wfc_ij_g = np.fft.fftn(wfc_ij, norm='forward') 
                    factor = 1.0
                    if ibnd_i != ibnd_j:
                        factor = 2.0
                    lower_chunk += factor * np.sum(wfc_ij * np.real(np.fft.ifftn(v_g * wfc_ij_g, norm='forward')) * occ[ispin, ibnd_j] * occ[ispin, ibnd_i] )
                    upper_chunk += factor * np.sum(wfc_ij * np.real(np.fft.ifftn(v_g_mu * wfc_ij_g, norm='forward')) * occ[ispin, ibnd_j] * occ[ispin, ibnd_i] )
                if rank == 0:
                    value = size
                    if nbnd[ispin] - ibnd_i < value:
                        value = nbnd[ispin] - ibnd_i
                    pbar.update(value)
        if rank == 0:
            pbar.close()


    lower = comm.allreduce(lower_chunk, op=MPI.SUM)
    upper = comm.allreduce(upper_chunk, op=MPI.SUM)

    if rank == 0:
        print(f"upper / lower: {upper/lower:5.3f}")
    read_obj.clean_wfc(storeFolder=storeFolder)


if __name__ == "__main__":
    n = 68
    while factorizable(n) is False:
        n += 2
    print(n)
