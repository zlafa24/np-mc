#!/usr/bin/python
import sys
import numpy as np
import string
from math import *
import random as rnd
from subprocess import call
from mc_library_rev3 import *
import time
from lammps import lammps


if __name__ == "__main__":
	T=298
	kb = 0.0019872041
	beta = 1/(kb*T)
	tries = 4000
	agID = 1
	ch2ID = 2
	ch3ID = 3
	sulfurID = 4
	oxygenID = 5
	hydrogenID = 6

	centerRotation = [40.9,40.9,40.]
	rotationtype = 'swap'	

	potentialfile = open('Potential_Energy.txt','w')
	potentialfile.write('Step\tPotential (kcal/mol)\n')

	max_angle = 0.34906585
	inputfile = 'addmolecule_184_rand.lmp'
	pe = getPotential('pe.out')
	atoms = readfile_c(inputfile)
	natoms = atoms.shape[0]
	molIDs = atoms[np.where(atoms[:,2]==sulfurID)][:,1]
	ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
	meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]	
	lmp = lammps()
	#lmp.file('addmolecule_184_rand.lmp')

	lmp.file("ddt_me_200.lmi")
	lmp.command("neigh_modify exclude type 1 1")
	coords = lmp.extract_atom("x",3)
	
	for i in xrange(tries):
		pe_old = pe
		coord_old = atoms[:,4:7]
		move = rnd.choice(['swap','rotate'])
		if(move=='swap'):
			ddtmol = rnd.choice(ddtMols)
			meohmol = rnd.choice(meohMols)
			swapMolecules(ddtmol,meohmol,atoms,centerRotation,rotationtype)
		elif(move=='rotate'):
			molId = rnd.choice(molIDs)
			randomRotate(atoms,molId,max_angle)
 	

		coords = atoms[:,4:7]
		print "Extracted coordinates are:"+str(coords[0][2])
		#lmp.put_coords(np.reshape(atoms[:,4:7],(1,natoms)))
		lmp.command("run 1 pre no post no")
		pe = lmp.extract_compute("thermo_pe",0,0)
		print "On loop: "+str(i)
		if((pe<pe_old) or (exp(-beta*(pe-pe_old))<rnd.uniform(0,1))):
			print "Move accepted"
		else:
			atoms[:,4:7] = coord_old
			pe = pe_old
		potentialfile.write(str(i)+'\t'+str(pe)+'\n')
	lmp.close()
