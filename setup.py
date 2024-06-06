# -*- coding: utf-8 -*-

from setuptools import find_packages, setup, Extension
import sys, subprocess
import pkg_resources

try:
    pkg_resources.require("Cython")
    from Cython.Build import cythonize
except pkg_resources.DistributionNotFound:
    print("Cython not found. Installing Cython...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
    from Cython.Build import cythonize

try:
    pkg_resources.require("Numpy")
    import numpy
except pkg_resources.DistributionNotFound:
    print("Numpy not found. Installing Numpy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy

# Define the extension module with the correct path
extensions = [
    Extension(
        "qcat.atomicEnv.kernel",                            # Module name
        ["./qcat/atomicEnv/kernel.pyx"],         # Cython source file
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        extra_compile_args=['-O3', '-march=native'],
    )
]

setup(
    name='qcat',
    version='1.6.0',
    ext_modules=cythonize(extensions, language_level="3"),
    packages=find_packages(),
    install_requires=[
        "pandas",
        "loguru",
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
