# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name='abinitioToolKit',
    version='1.5.0',
    packages=['abinitioToolKit',
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
        "PyYAML",
        "pyscf",
        "e3nn",
        ],
    include_package_data=True,
)
