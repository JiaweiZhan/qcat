from setuptools import setup, Extension
import sys, subprocess
import pkg_resources

try:
    pkg_resources.require("Cython")
    from Cython.Build import cythonize
except pkg_resources.DistributionNotFound:
    print("Cython not found. Installing Cython...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
    from Cython.Build import cythonize

import numpy

# Define the extension module with the correct path
extensions = [
    Extension(
        "kernel",                            # Module name
        ["./qcat/atomicEnv/kernel.pyx"],         # Cython source file
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        extra_compile_args=['-O3', '-march=native'],
    )
]

if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    package_dir={'': './qcat/atomicEnv/'},  # This tells setuptools that packages are under assignGrid
)
