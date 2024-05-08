# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup
import subprocess
import sys

args = [sys.executable, "./setup_ext.py"]
subprocess.check_call(args)

setup(
    name='qcat',
    version='1.6.0',
    packages=['qcat',
              ],
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
        "Cython",
        ],
    include_package_data=True,
)
