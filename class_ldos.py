import numpy as np
from tqdm import tqdm
import os, time
import threading
import qe_io
import shutil

class LDOS:
    # thread-related global value
    z_axis_assign = []
    lock_list = threading.Lock()
    condition_list = threading.Condition(lock_list)
    lcbm = None
    lvbm = None
    ksStateZAve = None
    numOcc = None               # [nks * nbnd]
    kWeights = None
    eigens = None               # [nks * nbnd]
    delta = None
    numThread = None
    nspin = None
    nks = None
    nbnd = None
    saveFolder = None
    storeFolder = None
    backup = None

    def __init__(self, delta=0.001, numThread=15, saveFolder='./scf.save', backup=1):
        """
        Init LDOS class:
            param:
                delta: float;
                numThread: int;
                saveFolder: str
                backp: bool
        """
        self.delta = delta
        self.numThread = numThread
        self.ksStateZAve = None
        self.lvbm = None 
        self.lcbm = None 
        self.saveFolder = saveFolder
        self.backup = backup

    def ldos_worker(self, id):
        """
        worker to compute LCBM and LVBM for multiple 
        z axis in z_axis_assign in a multi-threaded way
            param:
                id: int
        """
        self.lock_list.acquire()
        while len(self.z_axis_assign) == 0:
            self.condition_list.wait()
        z_axis = self.z_axis_assign.pop(0)
        # poison
        if len(z_axis) == 0:
            self.z_axis_assign.append([])
            self.lock_list.release()
            return
        # not a poison
        self.lock_list.release()

        if id == 0:
            for z in tqdm(z_axis, desc='compute LDOS'):
                # ksStateZAve: [ispin, ik, ibnd, z]
                preFactor = self.ksStateZAve[:, :, :, z]
                sumVBTot = np.sum(preFactor * self.numOcc * self.kWeights[np.newaxis, :, np.newaxis])

                KSEnergyTot = []
                KSFactorTot = []
                for i in range(self.nspin):
                    for j in range(self.nks):
                        KSEnergyTot.extend(self.eigens[i, j])
                        KSFactorTot.extend(preFactor[i, j] * self.kWeights[j])

                zipEneFac = zip(KSEnergyTot, KSFactorTot)
                eneSort, facSort = list(zip(*sorted(zipEneFac, key=lambda x: x[0])))

                min_arg = int(np.sum(self.numOcc)) - 1
                max_arg = int(np.sum(self.numOcc))

                sumLeft = 0
                while min_arg >= 0:
                    sumLeft += facSort[min_arg]
                    if sumLeft >= sumVBTot * self.delta:
                        break
                    else:
                        min_arg -= 1
                if min_arg != int(np.sum(self.numOcc)) -1:
                    min_arg += 1

                sumRight = 0
                while max_arg <= len(eneSort) - 1:
                    sumRight += facSort[max_arg]
                    if sumRight >= sumVBTot * self.delta:
                        break
                    else:
                        max_arg += 1
                if max_arg != int(np.sum(self.numOcc)):
                    max_arg -= 1
                self.lvbm[z] = eneSort[min_arg]
                self.lcbm[z] = eneSort[max_arg]
        else:
            for z in z_axis:
                # ksStateZAve: [ispin, ik, ibnd, z]
                preFactor = self.ksStateZAve[:, :, :, z]
                sumVBTot = np.sum(preFactor * self.numOcc * self.kWeights[np.newaxis, :, np.newaxis])

                KSEnergyTot = []
                KSFactorTot = []
                for i in range(self.nspin):
                    for j in range(self.nks):
                        KSEnergyTot.extend(self.eigens[i, j])
                        KSFactorTot.extend(preFactor[i, j] * self.kWeights[j])

                zipEneFac = zip(KSEnergyTot, KSFactorTot)
                eneSort, facSort = list(zip(*sorted(zipEneFac, key=lambda x: x[0])))

                min_arg = int(np.sum(self.numOcc)) - 1
                max_arg = int(np.sum(self.numOcc))

                sumLeft = 0
                while min_arg >= 0:
                    sumLeft += facSort[min_arg]
                    if sumLeft >= sumVBTot * self.delta:
                        break
                    else:
                        min_arg -= 1
                if min_arg != int(np.sum(self.numOcc)) -1:
                    min_arg += 1

                sumRight = 0
                while max_arg <= len(eneSort) - 1:
                    sumRight += facSort[max_arg]
                    if sumRight >= sumVBTot * self.delta:
                        break
                    else:
                        max_arg += 1
                if max_arg != int(np.sum(self.numOcc)):
                    max_arg -= 1
                self.lvbm[z] = eneSort[min_arg]
                self.lcbm[z] = eneSort[max_arg]
        return

    def computeLDOS(self):
        wfc_files = [self.saveFolder + '/' + file for file in os.listdir(self.saveFolder) if 'wfc' in file]

        # xml_data = utils.parse_QE_XML(xml_file)
        # # num of occupied bands
        # nks = xml_data['nks']
        # nspin = xml_data['nspin']
        # utils.numOcc = xml_data['occ']   # [nspin * nks * nbnds]
        # utils.kWeights = xml_data['kweights']
        # utils.tot_bands = xml_data['nbnd']
        # utils.eigens = xml_data['eigen'] # [nspin * nks * nbnds]
        # fft_grid = xml_data['fftw']
        # fft_grid = np.array(fft_grid) // 2 + 1
        qe = qe_io.QERead(numThread=self.numThread)
        xml_data = qe.parse_QE_XML(self.saveFolder + '/data-file-schema.xml')
        self.numOcc = xml_data['occ']
        self.kWeights = xml_data['kweights']
        self.nbnd = xml_data['nbnd']
        self.eigens = xml_data['eigen']
        self.nspin = self.numOcc.shape[0]
        self.nks = self.numOcc.shape[1]
        self.nbnd = self.numOcc.shape[2]
        fft_grid = xml_data['fftw']
        fft_grid = np.array(fft_grid) // 2 + 1

        self.ksStateZAve = np.zeros((self.nspin, self.nks, self.nbnd, fft_grid[2]))
        for index in tqdm(range(self.nks * self.nspin), desc='read wfc'):
            wfc_data = qe.parse_QE_wfc(wfc_files[index])
            ik = wfc_data['ik']
            ispin = wfc_data['ispin']
            # print(f"ik: {ik}, ispin: {ispin}, nbnd: {wfc_data['nbnd']}, npol: {wfc_data['npol']}, igwx: {wfc_data['igwx']}")

            # store and read
            if self.backup == 1:
                storeFolder = wfc_files[index].split('/')[-1].split('.')[0]
                self.storeFolder = storeFolder
                qe.computeWFC(Store=True, storeFolder=storeFolder)
                wfcStored = os.listdir(storeFolder)
                # for fileName in tqdm(wfcStored, desc='read wfc from stored file'):
                for fileName in wfcStored:
                    wfcName = storeFolder + '/' + fileName
                    ibnd = int(fileName.split('.')[0].split('_')[-1])
                    evc_r = np.load(wfcName)
                    self.ksStateZAve[ispin - 1, ik - 1, ibnd - 1, :] = np.sum(np.absolute(evc_r) ** 2, axis=(0, 1,))
                shutil.rmtree(storeFolder)
            else:
                # direct comput and store in memory
                evc_r = qe.computeWFC(Store=False)
                self.ksStateZAve[ispin - 1, ik - 1, :, :] = np.sum(np.absolute(evc_r) ** 2, axis=(1, 2,))

        self.lcbm = np.zeros(fft_grid[2])
        self.lvbm = np.zeros(fft_grid[2])
        #define lock and condition
        self.lock_list = threading.Lock()
        self.condition_list = threading.Condition(self.lock_list)

        num_thread = self.numThread
        length_chunk = fft_grid[2] / num_thread
        a_axis_tot = list(range(fft_grid[2]))

        self.lock_list.acquire()
        for i in range(num_thread):
            self.z_axis_assign.append(a_axis_tot[int(i * length_chunk) : int((i + 1) * length_chunk)])

        # poison
        self.z_axis_assign.append([])
        self.condition_list.notify_all()
        self.lock_list.release()

        # start thread
        threads = []
        ids = []
        for i in range(num_thread):
            ids.append(i)
            thread_id = threading.Thread(target=self.ldos_worker, args = (ids[i],))
            threads.append(thread_id)

        for thread_id in threads:
            thread_id.start()

        for thread_id in threads:
            thread_id.join()

    def localBandEdge(self):
        return self.lcbm, self.lvbm

if __name__=="__main__":
    # get the start time
    st = time.time()

    numThread = 15
    qe = qe_io.QERead(numThread)
    qe.parse_QE_XML('../si_si3n4_gamma.save/data-file-schema.xml')
    qe.parse_QE_wfc('../si_si3n4_gamma.save/wfc1.dat')
    qe.computeWFC(realSpace=True, Store=True, storeFolder='./wfc1/')
    # print(wfc_data["ngw"])
    # print(xml_data["occ"])

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
