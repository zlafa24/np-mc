from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("mc_library_rev3.pyx"))
