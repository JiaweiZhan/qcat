from xml.dom import minidom
import numpy as np
import datetime
from tqdm import tqdm
import os, time
import threading

def parse_QE_XML(file_name):
    """
    analyze QE data-file-schema.xml
        param:
            file_name: str
        return:
             dict of {'nks', 'kweights', 'nbnd', 'eigen', 'occ', 'fftw'}
    """
    # unit convert
    hartree2ev = 27.2114
    file = minidom.parse(file_name)
    # spin
    spin_nodes = file.getElementsByTagName('spin')
    lsda_nodes = spin_nodes[0].getElementsByTagName('lsda')
    nspin = 1
    if lsda_nodes[0].firstChild.data == 'true':
        nspin = 2


    # num of KS states
    nbnd_up, nbnd_dw = -1, -1
    nbnd_nodes = file.getElementsByTagName('nbnd')
    nbnd = int(nbnd_nodes[0].firstChild.data)
    if nspin == 2:
        nbnd_up_nodes = file.getElementsByTagName('nbnd_up')
        nbnd_up = int(nbnd_up_nodes[0].firstChild.data)
        nbnd_dw_nodes = file.getElementsByTagName('nbnd_dw')
        nbnd_dw = int(nbnd_dw_nodes[0].firstChild.data)
        assert(nbnd == nbnd_dw and nbnd == nbnd_up)

    # fermi energy
    fermi_nodes = file.getElementsByTagName('fermi_energy')
    fermiEne = float(fermi_nodes[0].firstChild.data) * hartree2ev

    # num of kpoints
    nks_nodes = file.getElementsByTagName('nks')
    nks = int(nks_nodes[0].firstChild.data)
 
    # [nspin * nks * nbnd]
    eigenvalues, occupations, weights = np.zeros((nspin, nks, nbnd)), np.zeros((nspin, nks, nbnd)), np.zeros(nks)

    ks_nodes = file.getElementsByTagName('ks_energies')
    for index, ks_node in enumerate(ks_nodes):
        # kpoint weight
        k_point_node = ks_node.getElementsByTagName('k_point')
        weights[index] = float(k_point_node[0].attributes['weight'].value)

        # eigenvalues
        eigens_node = ks_node.getElementsByTagName('eigenvalues')
        eigenvalue_ = eigens_node[0].firstChild.data
        eigenvalue_ = [float(num) * hartree2ev for num in eigenvalue_.split()]
        for ispin in range(nspin):
            eigenvalues[ispin, index, :] = eigenvalue_[ispin * nbnd : (ispin + 1) * nbnd]

        # occupation
        # occ_nodes = ks_node.getElementsByTagName('occupations')
        # occupation_ = occ_nodes[0].firstChild.data
        # occupation_ = [float(num) for num in occupation_.split()]
        # for ispin in range(nspin):
        #     occupations[ispin, index, :] = occupation_[ispin * nbnd : (ispin + 1) * nbnd]

        # occupation another way
        occupations[eigenvalues <= fermiEne] = 1
        occupations[eigenvalues > fermiEne] = 0

    # fft grids
    fft_nodes = file.getElementsByTagName('fft_grid')
    np0v = int(fft_nodes[0].attributes['nr1'].value)
    np1v = int(fft_nodes[0].attributes['nr2'].value)
    np2v = int(fft_nodes[0].attributes['nr3'].value)
    out_dict = {'nspin': nspin,
                'nks': nks,
                'fermi': fermiEne,
                'kweights': weights,
                'nbnd': nbnd,
                'eigen': eigenvalues,
                'occ': occupations,
                'fftw': [np0v, np1v, np2v]}
    return out_dict 

def parse_QE_wfc(file_name):
    """
    analyze QE wfc*.dat 
        param:
            file_name: str
        return:
             dict of {'ik', 'xk', 'ispin', 'nbnd', 'ngw', 'evc'...}
    """
    with open(file_name, 'rb') as f:
        # Moves the cursor 4 bytes to the right
        f.seek(4)

        ik = np.fromfile(f, dtype='int32', count=1)[0]
        xk = np.fromfile(f, dtype='float64', count=3)
        ispin = np.fromfile(f, dtype='int32', count=1)[0]
        gamma_only = bool(np.fromfile(f, dtype='int32', count=1)[0])
        scalef = np.fromfile(f, dtype='float64', count=1)[0]

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        ngw = np.fromfile(f, dtype='int32', count=1)[0]
        igwx = np.fromfile(f, dtype='int32', count=1)[0]
        npol = np.fromfile(f, dtype='int32', count=1)[0]
        nbnd = np.fromfile(f, dtype='int32', count=1)[0]

        # Move the cursor 8 byte to the right
        f.seek(8, 1)

        b1 = np.fromfile(f, dtype='float64', count=3)
        b2 = np.fromfile(f, dtype='float64', count=3)
        b3 = np.fromfile(f, dtype='float64', count=3)

        f.seek(8,1)
        
        mill = np.fromfile(f, dtype='int32', count=3*igwx)
        mill = mill.reshape( (igwx, 3) ) 

        evc = np.zeros( (nbnd, npol*igwx), dtype="complex128")
        f.seek(8,1)
        for i in tqdm(range(nbnd), desc='read wfc bands'):
            evc[i,:] = np.fromfile(f, dtype='complex128', count=npol*igwx)
            f.seek(8, 1)

        wfc_dict = {'ik': ik,
                    'xk': xk,
                    'ispin': ispin,
                    'gamma_only': gamma_only,
                    'scalef': scalef,
                    'ngw': ngw,
                    'igwx': igwx,
                    'npol': npol,
                    'nbnd': nbnd,
                    'mill': mill,
                    'b': [b1, b2, b3],
                    'evc': evc}
    return wfc_dict

# thread-related global value
z_axis_assign = []
lcbm = lvbm = None
lock_list = condition_list = None
ksStateZAve = None
numOcc = None               # [nks * nbnds]
kWeights = None
tot_bands = None
eigens = None               # [nks * nbnds]
delta = None

def ldos_worker(id):
    """
    worker to compute LCBM and LVBM for multiple 
    z axis in z_axis_assign in a multi-threaded way
        param:
            id: int
    """
    lock_list.acquire()
    while len(z_axis_assign) == 0:
        condition_list.wait()
    z_axis = z_axis_assign.pop(0)
    # poison
    if len(z_axis) == 0:
        z_axis_assign.append([])
        lock_list.release()
        return
    # not a poison
    lock_list.release()

    if id == 0:
        for z in tqdm(z_axis, desc='compute LDOS'):
            # ksStateZAve: [ispin, ik, ibnd, z]
            preFactor = ksStateZAve[:, :, :, z]
            sumVBTot = np.sum(preFactor * numOcc * kWeights[np.newaxis, :, np.newaxis])

            KSEnergyTot = []
            KSFactorTot = []
            for i in range(eigens.shape[0]):
                for j in range(eigens.shape[1]):
                    KSEnergyTot.extend(eigens[i, j])
                    KSFactorTot.extend(preFactor[i, j] * kWeights[j])

            zipEneFac = zip(KSEnergyTot, KSFactorTot)
            eneSort, facSort = list(zip(*sorted(zipEneFac, key=lambda x: x[0])))

            min_arg = int(np.sum(numOcc)) - 1
            max_arg = int(np.sum(numOcc))

            sumLeft = 0
            while min_arg >= 0:
                sumLeft += facSort[min_arg]
                if sumLeft >= sumVBTot * delta:
                    break
                else:
                    min_arg -= 1
            if min_arg != int(np.sum(numOcc)) -1:
                min_arg += 1

            sumRight = 0
            while max_arg <= len(eneSort) - 1:
                sumRight += facSort[max_arg]
                if sumRight >= sumVBTot * delta:
                    break
                else:
                    max_arg += 1
            if max_arg != int(np.sum(numOcc)):
                max_arg -= 1
            lvbm[z] = eneSort[min_arg]
            lcbm[z] = eneSort[max_arg]
    else:
        for z in z_axis:
            # ksStateZAve: [ispin, ik, ibnd, z]
            preFactor = ksStateZAve[:, :, :, z]
            sumVBTot = np.sum(preFactor * numOcc * kWeights[np.newaxis, :, np.newaxis])

            KSEnergyTot = []
            KSFactorTot = []
            for i in range(eigens.shape[0]):
                for j in range(eigens.shape[1]):
                    KSEnergyTot.extend(eigens[i, j])
                    KSFactorTot.extend(preFactor[i, j] * kWeights[j])

            zipEneFac = zip(KSEnergyTot, KSFactorTot)
            eneSort, facSort = list(zip(*sorted(zipEneFac, key=lambda x: x[0])))

            min_arg = int(np.sum(numOcc)) - 1
            max_arg = int(np.sum(numOcc))

            sumLeft = 0
            while min_arg >= 0:
                sumLeft += facSort[min_arg]
                if sumLeft >= sumVBTot * delta:
                    break
                else:
                    min_arg -= 1
            if min_arg != int(np.sum(numOcc)) -1:
                min_arg += 1

            sumRight = 0
            while max_arg <= len(eneSort) - 1:
                sumRight += facSort[max_arg]
                if sumRight >= sumVBTot * delta:
                    break
                else:
                    max_arg += 1
            if max_arg != int(np.sum(numOcc)):
                max_arg -= 1
            lvbm[z] = eneSort[min_arg]
            lcbm[z] = eneSort[max_arg]
    return

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

# shared variable
storeIbndList = []
lock_Ibnds = threading.Lock()
iBnds_condition = threading.Condition(lock_Ibnds)

def storeGvec(xml_data, wfc_data, realSpace=True, Store=False, storeFolder=None, threadNum=None):
    """
    worker to storeGVec in a multi-threaded way
    by either storing wfc in memory or disk
    param:
        xml_data: return of parse_QE_XML
        wfc_data: return of parse_QE_wfc
        realSpace: bool, whether convert wfc to real space
        Store: bool, whether store wfc on disk to save memory
        storeFolder: str, whether the wfc is strored
        threadNum: number of thread to store wfc
    """
    fft_grid = xml_data['fftw']
    fft_grid = np.array(fft_grid) // 2 + 1
    if not Store:
        # store everyting in memory
        evc_g = np.zeros([wfc_data['nbnd'], fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
        for index, gvec in enumerate(tqdm(wfc_data['mill'], desc='store wfc_g')):
            eig = wfc_data['evc'][:, index]
            evc_g[:, gvec[0], gvec[1], gvec[2]] = eig
            if (gvec[0] != 0 or gvec[1] != 0 or gvec[2] != 0):
                evc_g[:, -gvec[0], -gvec[1], -gvec[2]] = np.conj(eig)
        if not realSpace:
            return evc_g
        else:
            evc_r = np.fft.ifftn(evc_g, axes=(1, 2, 3,), norm='forward')
            return evc_r
    else:
        # memory is not enough, store file under folder for further loading
        assert(storeFolder is not None)
        isExist = os.path.exists(storeFolder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(storeFolder)
        millIndex = wfc_data['mill']
        allZeroIndex = -1
        for index in range(millIndex.shape[0]):
            if millIndex[index, 0] == 0 and millIndex[index, 1] == 0 and millIndex[index, 2] == 0:
                allZeroIndex = index
                break
        if allZeroIndex != -1:
            millIndex = np.delete(millIndex, allZeroIndex, axis=0)

        lock_Ibnds.acquire()
        global storeIbndList
        storeIbndList = list(range(wfc_data['nbnd']))
        storeIbndList.append(-1)
        iBnds_condition.notify_all()
        lock_Ibnds.release()
        
        threads, ids = [], []
        for i in range(threadNum):
            ids.append(i)
            thread_id = threading.Thread(target=storeGWorker, args=(wfc_data, millIndex, fft_grid, allZeroIndex, storeFolder, realSpace, ids[i],))
            threads.append(thread_id)

        for thread_id in threads:
            thread_id.start()

        for thread_id in threads:
            thread_id.join()
        
        # TODO: multithread
        # for ibnd in tqdm(range(wfc_data['nbnd']), desc='store ibnd'):
        #     evc_g = np.zeros([fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
        #     wfc_slice = list(wfc_data['evc'][ibnd, :])
        #     evc_g[wfc_data['mill'][:, 0], wfc_data['mill'][:, 1], wfc_data['mill'][:, 2]] = wfc_data['evc'][ibnd, :]
        #     if allZeroIndex != -1:
        #         del wfc_slice[allZeroIndex]
        #     evc_g[ - millIndex[:, 0], - millIndex[:, 1], - millIndex[:, 2]] = np.conj(wfc_slice)
        #     evc_r = None
        #     if not realSpace:
        #         wfcName = 'wfc_g_' + str(ibnd + 1).zfill(5)
        #         np.save(storeFolder + '/' + wfcName, evc_g)
        #     else:
        #         evc_r = np.fft.ifftn(evc_g, norm='forward')
        #         wfcName = 'wfc_r_' + str(ibnd + 1).zfill(5)
        #         np.save(storeFolder + '/' + wfcName, evc_r)
        return

def storeGWorker(wfc_data, millIndex, fft_grid, allZeroIndex, storeFolder, realSpace, id):
    while True:
        lock_Ibnds.acquire()
        while len(storeIbndList) == 0:
            iBnds_condition.wait()
        iBnd = storeIbndList.pop(0)
        # poison
        if iBnd == -1:
            storeIbndList.append(-1)
            lock_Ibnds.release()
            return
        # not a poison
        lock_Ibnds.release()

        if id == 0:
            # for ibnd in tqdm(iBnds, desc='store ibnd'):
            evc_g = np.zeros([fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
            wfc_slice = wfc_data['evc'][iBnd, :]
            evc_g[wfc_data['mill'][:, 0], wfc_data['mill'][:, 1], wfc_data['mill'][:, 2]] = wfc_data['evc'][iBnd, :]
            if allZeroIndex != -1:
                wfc_slice = np.delete(wfc_slice, allZeroIndex)
            evc_g[- millIndex[:, 0], - millIndex[:, 1], - millIndex[:, 2]] = np.conj(wfc_slice)
            evc_r = None
            if not realSpace:
                wfcName = 'wfc_g_' + str(iBnd + 1).zfill(5)
                np.save(storeFolder + '/' + wfcName, evc_g)
            else:
                evc_r = np.fft.ifftn(evc_g, norm='forward')
                wfcName = 'wfc_r_' + str(iBnd + 1).zfill(5)
                np.save(storeFolder + '/' + wfcName, evc_r)
        else:
            evc_g = np.zeros([fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
            wfc_slice = wfc_data['evc'][iBnd, :]
            evc_g[wfc_data['mill'][:, 0], wfc_data['mill'][:, 1], wfc_data['mill'][:, 2]] = wfc_data['evc'][iBnd, :]
            if allZeroIndex != -1:
                wfc_slice = np.delete(wfc_slice, allZeroIndex)
            evc_g[- millIndex[:, 0], - millIndex[:, 1], - millIndex[:, 2]] = np.conj(wfc_slice)
            evc_r = None
            if not realSpace:
                wfcName = 'wfc_g_' + str(iBnd + 1).zfill(5)
                np.save(storeFolder + '/' + wfcName, evc_g)
            else:
                evc_r = np.fft.ifftn(evc_g, norm='forward')
                wfcName = 'wfc_r_' + str(iBnd + 1).zfill(5)
                np.save(storeFolder + '/' + wfcName, evc_r)

if __name__=="__main__":
    # get the start time
    st = time.time()

    xml_data = parse_QE_XML('../si_si3n4.save/data-file-schema.xml')
    wfc_data = parse_QE_wfc('../si_si3n4.save/wfc1.dat')
    storeGvec(xml_data, wfc_data, realSpace=False, Store=True, storeFolder='./wfc1/', threadNum=20)
    # print(wfc_data["ngw"])
    # print(xml_data["occ"])

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
