
NP-MC
=====

LAMMPS Monte Carlo for Nanoparticle Monolayer
----------------------------------------------

np-mc is Python library designed to make building Monte Carlo simulations for nanoparticle monolayers easier.  The program uses LAMMPS to evaluate energies of the system and uses a simple object oriented library to modify the underlying system according to extendable Monte Carlo moves.  

Installation
------------

### Installing LAMMPS with Python Wrapper


The np-mc package depends on the LAMMPS Python wrapper which needs to be installed before running any np-mc scripts with calls to LAMMPS (any using the Simulation or Move class).  To install LAMMPS with the Python wrapper first download the desired version of LAMMPS from github:

```
git clone -b ${DESIRED_VERSION_TAG} https://github.com/lammps/lammps.git
```

Once LAMMPS has finished cloning the repository to your local directory make a `build` directory in the `lammps` directory to use to store the build files for LAMMPS:

```
cd lammps
mkdir build
```

Once in the build directory the LAMMPS Python wrapper can be built by compiling LAMMPS as a shared library with the PYTHON package `-D PACKAGE_PYTHON=on`.  NP-MC also takes advantage of the MOLECULE package `-D PACKAGE_MOLECULE=on`.  To create the configuration file run:

```
cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D PKG_PYTHON=on -D PKG_MOLECULE=on -D BUILD_LIB=on -D BUILD_SHARED_LIBS=on ../cmake

```

If any additional LAMMPS packages are desired they can be added using the cmake option `-D PACKAGE_${PACKAGE_NAME}` where `${PACKAGE_NAME}` is the name of the desired package.  Once cmake completes compile the LAMMPS using the make command:

```
make -j 4
```

Once compiled the LAMMPS library can be installed with make using:

```
sudo make install
```


### Installing NP-MC


np-mc can be installed using:

```
python setup.py
```

or using the PIP package manager using:

```
pip install .
```

