#!/scratch/midway2/jiaweiz/anaconda3/bin/python3
import numpy as np
from tqdm import tqdm
import argparse
import utils
import threading

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wfc_file", type=str,
            help="filename that include wfc. Default: wfc1.dat")
    parser.add_argument("-x", "--xml_file", type=str,
            help="filename that include xml data. Default: data-file-schema.xml")
    parser.add_argument("-d", "--delta", type=float,
            help="delta that control local VB/CB. Default: 0.001")
    parser.add_argument("-t", "--thread", type=int,
            help="num of thread. Default: 15")
    args = parser.parse_args()

    # default values
    if not args.wfc_file:
        args.wfc_file = "wfc1.dat"
    if not args.xml_file:
        args.xml_file = "data-file-schema.xml"
    if not args.thread:
        args.thread = 15 
    if not args.delta:
        args.delta = 0.001 

    utils.delta = args.delta

    print(f"configurations:\
            \n\t{'wfc file:':^15}{args.wfc_file:^20}\
            \n\t{'xml file:':^15}{args.xml_file:^20}\
            \n\t{'delta:':^15}{args.delta:^20}\
            \n\t{'threads:':^15}{args.thread:^20}\
            ")

    # -----------------------------------------------------------------------------

    xml_data = utils.parse_QE_XML(args.xml_file)
    wfc_data = utils.parse_QE_wfc(args.wfc_file)
    wfc = wfc_data['evc']
    print(f"nbnd: {wfc_data['nbnd']}, npol: {wfc_data['npol']}, igwx: {wfc_data['igwx']}")

    # num of occupied bands
    utils.numOcc = int(np.sum(xml_data['occ']))
    utils.tot_bands = int(len(xml_data['occ']))
    utils.eigens = xml_data['eigen']

    fft_grid = xml_data['fftw']
    fft_grid = np.array(fft_grid) // 2 + 1
    evc_g = np.zeros([wfc_data['nbnd'], fft_grid[0], fft_grid[1], fft_grid[2]], dtype=np.complex128)
    index_x_m = index_y_m = index_z_m = -1
    last_x = last_y = last_z = last_x_m = last_y_m = last_z_m = -1
    for index, gvec in enumerate(tqdm(wfc_data['mill'], desc='store wfc_g')):
        eig = wfc_data['evc'][:, index]
        evc_g[:, gvec[0], gvec[1], gvec[2]] = eig
        if (gvec[0] != 0 or gvec[1] != 0 or gvec[2] != 0):
            evc_g[:, -gvec[0], -gvec[1], -gvec[2]] = np.conj(eig)

    utils.evc_r = np.fft.ifftn(evc_g, axes=(1, 2, 3,), norm='forward')

    utils.lcbm = np.zeros(fft_grid[2])
    utils.lvbm = np.zeros(fft_grid[2])
    #define lock and condition
    utils.lock_list = threading.Lock()
    utils.condition_list = threading.Condition(utils.lock_list)

    num_thread = args.thread
    length_chunk = fft_grid[2] / num_thread
    a_axis_tot = list(range(fft_grid[2]))

    utils.lock_list.acquire()
    for i in range(num_thread):
        utils.z_axis_assign.append(a_axis_tot[int(i * length_chunk) : int((i + 1) * length_chunk)])

    # poison
    utils.z_axis_assign.append([])
    utils.condition_list.notify_all()
    utils.lock_list.release()

    # start thread
    threads = []
    ids = []
    for i in range(num_thread):
        ids.append(i)
        thread_id = threading.Thread(target=utils.ldos_worker, args = (ids[i],))
        threads.append(thread_id)

    for thread_id in threads:
        thread_id.start()

    for thread_id in threads:
        thread_id.join()
    # thread end
    with open('ldos.txt', 'w') as file_object:
        file_object.writelines("LVBM:\n")
        for i in range(fft_grid[2]):
            file_object.write(f'{utils.lvbm[i]:12.5f}')
            if i % 5 == 4:
                file_object.write('\n')
        file_object.write('\n\n')
        file_object.writelines("LCBM:\n")
        for i in range(fft_grid[2]):
            file_object.write(f'{utils.lcbm[i]:12.5f}')
            if i % 5 == 4:
                file_object.write('\n')
