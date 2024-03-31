from abinitioToolKit import qe_io, qbox_io
import numpy as np
import os
from typing import Iterator, Tuple
from lxml import etree
from lxml.etree import _Element
from tqdm import tqdm
import base64
import pickle

class QE2Qbox(object):
    def __init__(self,
                 qbox_folder: str,
                 qe_folder: str,
                 workdir = None):
        self.qbox_folder = qbox_folder
        self.qe_folder = qe_folder
        workdir = 'wfc' if workdir is None else workdir
        self.workdir = os.path.join(os.getcwd(), workdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

    def extract_qe_data(self):
        storeFolder=os.path.join(self.workdir, "qe_wfc")
        qe_data = qe_io.QERead(outFolder=self.qe_folder)
        qe_data.parse_info(storeFolder=storeFolder)
        qe_data.parse_wfc(real_space=False,
                          storeFolder=storeFolder)
        return qe_data

    def extract_qbox_data(self):
        storeFolder=os.path.join(self.workdir, "qbox_wfc")
        qbox_data = qbox_io.QBOXRead(outFolder=self.qbox_folder)
        qbox_data.parse_info()
        qbox_data.parse_wfc(storeFolder=storeFolder,
                            store_wfc=False)
        return qbox_data

    @staticmethod
    def replace_wfc(qbox_xml: str,
                    qe_wfc_folder: str,
                    fftw: np.ndarray,
                    nbnd: np.ndarray,
                    output_file: str):
        '''
        the wfc in qbox format is stored in order [z, y, x] in real space
        '''
        assert os.path.exists(qe_wfc_folder), f"qe wfc folder {qe_wfc_folder} does not exist"
        assert os.path.exists(qbox_xml), f"qbox xml file {qbox_xml} does not exist"
        iwfc = 0
        ispin = 0

        total_iter = np.sum(nbnd)
        pbar = tqdm(desc='replace wfc', total=total_iter)

        mill = np.load(os.path.join(qe_wfc_folder, 'mill.npy'))
        mill_x, mill_y, mill_z = mill.T
        tree = etree.parse(qbox_xml)
        root = tree.getroot()
        for element in root.iter():
            if element.tag == "grid_function":
                 encoding = element.get("encoding")
                 qe_wfc_file = os.path.join(qe_wfc_folder, 'wfc_' + str(ispin + 1) + '_' + str(1).zfill(3) + '_' + str(iwfc + 1).zfill(5) + '_g.npy')
                 qe_wfc = np.load(qe_wfc_file)
                 wfc_g = np.zeros(fftw, dtype=np.complex128)
                 wfc_g[mill_x, mill_y, mill_z] = qe_wfc
                 wfc_g[-mill_x, -mill_y, -mill_z] = qe_wfc.conj()
                 wfc_r = np.fft.ifftn(wfc_g, norm='forward').real
                 wfc_r = np.transpose(wfc_r, (2, 1, 0))
                 if encoding.strip() == "text":
                     element.text = ' '.join(wfc_r.flatten().astype(str))
                 else:
                     element.text = base64.encodebytes(wfc_r.tobytes()).decode('utf-8')
                 pbar.update(1)
                 iwfc = (iwfc + 1) % nbnd[ispin]
                 if iwfc == 0:
                     ispin += 1
        tree.write(output_file, pretty_print=True, xml_declaration=True, encoding='utf-8')


    def run(self):
        qe_data = self.extract_qe_data()
        qbox_data = self.extract_qbox_data()

        qe_wfc_folder = os.path.join(self.workdir, "qe_wfc")
        qbox_wfc_folder = os.path.join(self.workdir, "qbox_wfc")

        qbox_xml_fname = str(qbox_data.xmlSample)
        qbox_wfc_dict = pickle.load(open(os.path.join(qbox_wfc_folder, "info.pickle"), "rb"))

        self.replace_wfc(qbox_xml_fname,
                         os.path.join(self.workdir, "qe_wfc"),
                         qbox_wfc_dict['fftw'],
                         qbox_wfc_dict['nbnd'],
                         os.path.join(self.workdir, "new_qbox.xml"))


        qe_data.clean_wfc(qe_wfc_folder)
        qbox_data.clean_wfc(qbox_wfc_folder)
