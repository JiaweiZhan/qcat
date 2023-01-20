# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='abinitioToolKit',
    version='0.0',
    packages=['abinitioToolKit',
              'abinitioToolKit/io.py',
              'abinitioToolKit/qe_io.py',
              'abinitioToolKit/utils.py',
              'abinitioToolKit/qbox_io.py',
              'abinitioToolKit/class_lf.py',
              'abinitioToolKit/class_ldos.py',
              ],
    # find_packages(exclude=[]),
    install_requires=[
        "h5py",
        "lxml",
        "matplotlib",
        "mpi4py",
        "numpy",
        "scipy",
        "tqdm",
        ],
    include_package_data=True,
)
