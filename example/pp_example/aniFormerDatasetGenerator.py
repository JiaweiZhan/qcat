# only support qbox yet
from qcat.io_kernel import qbox_io
from qcat.utils import utils
from mpi4py import MPI
import argparse
from functools import partial
import signal
import pickle
from tqdm import tqdm
import numpy as np
import shutil, os, yaml
import pickle, h5py, sys

comm = MPI.COMM_WORLD
bohr2angstrom = 0.529177249

PERIODIC_TABLE = """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()

def visualize_alpha(alpha, fileName = 'alpha_zoom.dat'):
    utils.visualize_func(alpha, zoom_factor=1.0, fileName=fileName)

def test_dataset():
    print("Runing TestSet")
    dataset_folder = './Dataset_test'
    structure_folder = 'structures'
    attribute_folder = 'attributes'
    material_name = 'test'
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    if not os.path.exists(os.path.join(dataset_folder, structure_folder)):
        os.mkdir(os.path.join(dataset_folder, structure_folder))
    if not os.path.exists(os.path.join(dataset_folder, attribute_folder)):
        os.mkdir(os.path.join(dataset_folder, attribute_folder))
    if not os.path.exists(os.path.join(dataset_folder, attribute_folder, material_name)):
        os.mkdir(os.path.join(dataset_folder, attribute_folder, material_name))

    structure_data = {
            'cell': [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
            'pbc': [True, True, True],
            'atomic_positions': [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            'species': [1, 6, 8],
            'species_order': ['H', 'C', 'O'],
            'target_point' : [
                [0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.8, 0.0, 0.0],
                [0.8, 0.8, 0.0],
                ]
            }
    structure_fname = f'test.pkl'
    with open(os.path.join(dataset_folder, structure_folder, structure_fname), 'wb') as pickle_file:
        pickle.dump(structure_data, pickle_file)
    attribute_fname = os.path.join(dataset_folder, attribute_folder, material_name, f"{material_name}__{1}")
    np.save(attribute_fname, np.array(5))
    attribute_fname = os.path.join(dataset_folder, attribute_folder, material_name, f"{material_name}__{2}")
    np.save(attribute_fname, np.array(5))
    attribute_fname = os.path.join(dataset_folder, attribute_folder, material_name, f"{material_name}__{3}")
    np.save(attribute_fname, np.array(10))
    attribute_fname = os.path.join(dataset_folder, attribute_folder, material_name, f"{material_name}__{4}")
    np.save(attribute_fname, np.array(10))

def outputStructure(structure_data, threeDGrid, point_gap, dataset_folder, structure_folder, material_name):
    divisions = np.array([1 / threeDGrid[0], 1 / threeDGrid[1], 1 / threeDGrid[2]], dtype=float)
    i, j, k = np.mgrid[0:threeDGrid[0]:point_gap, 0:threeDGrid[1]:point_gap, 0:threeDGrid[2]:point_gap]
    point_indices = np.array([i, j, k]).T.reshape(-1, 3)
    target_point = point_indices * divisions[None, :]
    structure_data['target_point'] = target_point.tolist()

    structure_fname = f'{material_name}.pkl'
    with open(os.path.join(dataset_folder, structure_folder, structure_fname), 'wb') as pickle_file:
        pickle.dump(structure_data, pickle_file)
    point_indices = point_indices.tolist()
    return point_indices

if __name__ == "__main__":
    # signal.signal(signal.SIGINT, partial(utils.handler, comm))

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        utils.time_now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--saveFileFolder", type=str,
            help="folder that store XML sample and qbox.out. Default: ../")
    parser.add_argument("-a", "--alphaFile", type=str,
            help="Local Dielectric Function File. Default: ../alpha.txt")
    parser.add_argument("-n", "--material_name", type=str,
            help="material_name. Default: sihwat")
    parser.add_argument("-o", "--species_order", nargs='*',
            help="species_order. Default: H O Si")
    parser.add_argument("-t", "--test", default=False, action='store_true',
            help="whether to run testset. Default: False")
    parser.add_argument("-v", "--visualize_alpha", default=False, action='store_true',
            help="whether to visualize alpha. Default: False")
    parser.add_argument("-e", "--eval_data", default=False, action='store_true',
            help="whether to output structure for evaluation of aniformer. Default: False")
    parser.add_argument("-oa", "--outputAttribute", type=str, 
            help="output 'rho' or 'epsilon'. Default: 'epsilon'")
    args = parser.parse_args()

    if args.test:
        if rank == 0:
            test_dataset()
        comm.Barrier()
        sys.exit(0)

    if not args.saveFileFolder:
        args.saveFileFolder = "../" 
    if not args.alphaFile:
        args.alphaFile = "../alpha.txt" 
    if not args.material_name:
        args.material_name = "sihwat" 
    if not args.species_order:
        args.species_order =  ['H', 'O', 'Si']
    if not args.outputAttribute:
        args.outputAttribute = "epsilon"
    args.outputAttribute = args.outputAttribute.strip().lower()
    if args.outputAttribute != "rho" and args.outputAttribute != "epsilon":
        raise KeyError("outputAttribute can only be epsilon or rho")
    material_name = args.material_name

    conf_tab = {"saveFileFolder": args.saveFileFolder,
                "alphaFile": args.alphaFile,
                "material_name": args.material_name,
                "species_order": args.species_order,
                "output_attribute": args.outputAttribute,
                "testset": args.test,
                "visualize": args.visualize_alpha,
                "eval_data": args.eval_data,
                "MPI size": comm.Get_size()}
    utils.print_conf(conf_tab)

    # ------------------------------------------- read and store wfc --------------------------------------------
    
    qbox = qbox_io.QBOXRead(comm=comm)
    storeFolder = './wfc/'

    comm.Barrier()
    isExist = os.path.exists(storeFolder)
    if not isExist:
        if rank == 0:
            print(f"store wfc from {storeFolder}")
        qbox.read(args.saveFileFolder, storeFolder=storeFolder)
    else:
        if rank == 0:
            print(f"read stored wfc from {storeFolder}")
     
    # --------------------------------------- generate training data for aniformer ----------------------------------------
    
    # comm.Barrier()
    with open(storeFolder + '/info.pickle', 'rb') as handle:
        info_data = pickle.load(handle)

    npv = info_data['npv']
    fftw = info_data['fftw']
    cell = info_data['cell'] * bohr2angstrom
    volume = np.abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))
    species_loc = info_data['atompos']
    nbnd = info_data['nbnd']
    nspin = info_data['nspin']
    occ = info_data['occ']
    fileNameList = info_data['wfc_file']

    name2index = {s: k for k, s, in enumerate(PERIODIC_TABLE, 1)}
    species, positions = [], []
    for i in species_loc:
        species.append(name2index[i[0]])
        positions.append(i[1:])
    positions = (np.array(positions) * bohr2angstrom).tolist()

    if args.eval_data:
        if rank == 0:
            structure_data = {
                    'cell': cell.tolist(),
                    'pbc': [True, True, True],
                    'atomic_positions': positions,
                    'species': species,
                    'species_order': args.species_order,
                    }

            # output only the structure_data without target_point
            eval_dataset_folder = './Dataset_eval'
            structure_folder = 'structures'
            attribute_folder = 'attributes'
            if not os.path.exists(eval_dataset_folder):
                os.mkdir(eval_dataset_folder)
            if not os.path.exists(os.path.join(eval_dataset_folder, structure_folder)):
                os.mkdir(os.path.join(eval_dataset_folder, structure_folder))
            structure_fname = f'{material_name}.pkl'
            with open(os.path.join(eval_dataset_folder, structure_folder, structure_fname), 'wb') as pickle_file:
                pickle.dump(structure_data, pickle_file)
            sys.exit(0)
        else:
            sys.exit(0)

    alpha = None
    threeDGrid = npv
    point_gap = 5
    if args.outputAttribute == 'epsilon':
        alphaFile = args.alphaFile 
        alpha = utils.read_alpha(alphaFile=alphaFile, npv=npv)
    else:
        point_gap = 5
        alphaFile = args.alphaFile 
        alpha = np.load(alphaFile)
        threeDGrid = alpha.shape
    if args.visualize_alpha:
        visualize_alpha(alpha)
        comm.Barrier()
        sys.exit(0)

    dataset_folder = './Dataset'
    structure_folder = 'structures'
    attribute_folder = 'attributes'
    if rank == 0:
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        if not os.path.exists(os.path.join(dataset_folder, structure_folder)):
            os.mkdir(os.path.join(dataset_folder, structure_folder))
        if not os.path.exists(os.path.join(dataset_folder, attribute_folder)):
            os.mkdir(os.path.join(dataset_folder, attribute_folder))
        if not os.path.exists(os.path.join(dataset_folder, attribute_folder, material_name)):
            os.mkdir(os.path.join(dataset_folder, attribute_folder, material_name))

    comm.Barrier()

    if rank == 0:
        structure_data = {
                'cell': cell.tolist(),
                'pbc': [True, True, True],
                'atomic_positions': positions,
                'species': species,
                'species_order': args.species_order,
                }
        point_indices = outputStructure(structure_data, threeDGrid, point_gap, dataset_folder, structure_folder, material_name)
    else:
        point_indices = None

    point_indices = comm.bcast(point_indices, root=0)
    point_indices = np.array(point_indices)

    if rank == 0:
        pbar = tqdm(desc=f'store attributes: ', total=point_indices.shape[0])
    comm.Barrier()

    i_indices, j_indices, k_indices = point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]
    attributes = alpha[i_indices, j_indices, k_indices]

    for index in range(attributes.shape[0]):
        if index % size == rank:
            attribute_fname = os.path.join(dataset_folder, attribute_folder, material_name, f"{material_name}__{index + 1}")
            file = open(attribute_fname+'.npy', 'wb')
            np.save(file, attributes[index])
            file.close()

            if rank == 0:
                value = size
                if  attributes.shape[0] - index < value:
                    value = attributes.shape[0] - index
                pbar.update(value)

    comm.Barrier()
    if rank == 0:
        pbar.close()

