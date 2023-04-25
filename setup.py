# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='abinitioToolKit',
    version='0.0',
    packages=['abinitioToolKit',
              # 'abinitioToolKit/io',
              # 'abinitioToolKit/qe_io',
              # 'abinitioToolKit/utils',
              # 'abinitioToolKit/qbox_io',
              # 'abinitioToolKit/class_lf',
              # 'abinitioToolKit/class_ldos',
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
        "scikit-learn",
        "PyYAML"
        ],
    include_package_data=True,
)
