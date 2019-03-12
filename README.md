
NP-MC
=====

LAMMPS Monte Carlo for Nanoparticle Monolayer
----------------------------------------------

np-mc is Python library designed to make building Monte Carlo simulations for nanoparticle monolayers easier.  The program uses LAMMPS to evaluate energies of the system and uses a simple object oriented library to modify the underlying system according to extendable Monte Carlo moves.  

Installation
------------

###Installing LAMMPS with Python Wrapper


The np-mc package depends on the LAMMPS Python wrapper which needs to be installed before running any np-mc scripts with calls to LAMMPS (any using the Simulation or Move class).  To install LAMMPS with the Python wrapper first download the desired version of LAMMPS from github:

```
git clone -b ${DESIRED_VERSION_TAG} https://github.com/lammps/lammps.git
```

Once LAMMPS has finished cloning the repository to your local directory make a `build` directory in the `lammps` directory to use to store the build files for LAMMPS:

```
cd lammps
mkdir build
```

###Installing NP-MC


np-mc can be installed using:

```
python setup.py
```

or using the PIP package manager using:

```
pip install .
```

