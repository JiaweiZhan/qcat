from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Define the extension module with the correct path
extensions = [
    Extension(
        "kernel",                            # Module name
        ["./qcat/assignGrid/kernel.pyx"],         # Cython source file
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        extra_compile_args=['-O3', '-march=native'],
    )
]

if len(sys.argv) == 1:
    sys.argv.extend(["build_ext", "--inplace"])

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    package_dir={'': './qcat/assignGrid/'},  # This tells setuptools that packages are under assignGrid
)
