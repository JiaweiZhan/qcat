from xml.dom import minidom
import numpy as np
import datetime
from tqdm import tqdm

def parse_QE_XML(file_name):
    file = minidom.parse(file_name)
    eigens_node = file.getElementsByTagName('eigenvalues')
    eigenvalues = eigens_node[0].firstChild.data
    eigenvalues = [float(num) * 27.2114 for num in eigenvalues.split()]

    occ_nodes = file.getElementsByTagName('occupations')
    occupations = occ_nodes[1].firstChild.data
    occupations = [float(num) for num in occupations.split()]

    fft_nodes = file.getElementsByTagName('fft_grid')
    np0v = int(fft_nodes[0].attributes['nr1'].value)
    np1v = int(fft_nodes[0].attributes['nr2'].value)
    np2v = int(fft_nodes[0].attributes['nr3'].value)
    print(f"FFTW3: {[np0v, np1v, np2v]}")
    out_dict = {'eigen': eigenvalues,
                'occ': occupations,
                'fftw': [np0v, np1v, np2v]}
    return out_dict 

def parse_QE_wfc(file_name):
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
        for i in range(nbnd):
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
evc_r = None
numOcc = None
tot_bands = None
eigens = None
delta = None

def ldos_worker(id):
    # worker to compute LCBM and LVBM for multiple z axis in z_axis_assign
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
            # evc_r = [ibnd, x, y, z]
            preFactor = np.sum(np.absolute(evc_r[:, :, :, z]) ** 2, axis=(1, 2))
            sumVBTot = np.sum(preFactor[:numOcc])

            min_arg = numOcc - 1
            max_arg = numOcc

            sumLeft = 0
            while min_arg >= 0:
                sumLeft += preFactor[min_arg]
                if sumLeft >= sumVBTot * delta:
                    break
                else:
                    min_arg -= 1
            if min_arg != numOcc -1:
                min_arg += 1

            sumRight = 0
            while max_arg <= tot_bands - 1:
                sumRight += preFactor[max_arg]
                if sumRight >= sumVBTot * delta:
                    break
                else:
                    max_arg += 1
            if max_arg != numOcc:
                max_arg -= 1
            lvbm[z] = eigens[min_arg]
            lcbm[z] = eigens[max_arg]
    else:
        for z in z_axis:
            # evc_r = [ibnd, x, y, z]
            preFactor = np.sum(np.absolute(evc_r[:, :, :, z]) ** 2, axis=(1, 2))
            sumVBTot = np.sum(preFactor[:numOcc])

            min_arg = numOcc - 1
            max_arg = numOcc

            sumLeft = 0
            while min_arg >= 0:
                sumLeft += preFactor[min_arg]
                if sumLeft >= sumVBTot * delta:
                    break
                else:
                    min_arg -= 1
            if min_arg != numOcc -1:
                min_arg += 1

            sumRight = 0
            while max_arg <= tot_bands - 1:
                sumRight += preFactor[max_arg]
                if sumRight >= sumVBTot * delta:
                    break
                else:
                    max_arg += 1
            if max_arg != numOcc:
                max_arg -= 1
            lvbm[z] = eigens[min_arg]
            lcbm[z] = eigens[max_arg]
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


if __name__=="__main__":
    xml_data = parse_QE_XML('./data-file-schema.xml')
    wfc_data = parse_QE_wfc('./wfc1.dat')
    print(wfc_data["ngw"])
    print(xml_data["occ"])
