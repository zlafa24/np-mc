from distutils.core import setup
from Cython.Build import cythonize

if __name__ == "__main__":
    setup(ext_modules=cythonize("mc_library_rev3.pyx"))
