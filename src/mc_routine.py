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
import read_lmp_rev6 as rdlmp
import itertools as itt
from ctypes import *

def initializeMols(atoms,bonds):
	ch3ID = 3
	sulfurID = 4
	oxygenID = 5

	ddtsulfurs = [atom[0] for atom in atoms if (atom[2]==sulfurID and (atom[1] in ddtMols))]
	meohsulfurs = [atom[0] for atom in atoms if (atom[2]==sulfurID and (atom[1] in meohMols))]

	ddts = np.empty([ddtMols.shape[0],13])
	meohs = np.empty([meohMols.shape[0],5])

	for idx,(ddtsulfur,meohsulfur) in enumerate(itt.izip_longest(ddtsulfurs,meohsulfurs)):
		if(not (ddtsulfur==None)):
			ddts[idx,:] = rdlmp.getMoleculeAtoms(bonds,ddtsulfur)
		if(not (meohsulfur==None)):
			meohs[idx,:] = rdlmp.getMoleculeAtoms(bonds,meohsulfur)
	return (ddts,meohs)

def generate_vector(k,theta0,phi0,r0)


def cbmc(atoms,mol,beta,lmpptr):
	lmp = lammps(ptr=lmptr)
	molId = atoms[atoms[:,0]=mol[0]][0,1]
	startindex = rnd.choice(range(1,len(mol)))
	atoms2del = mol[startindex:]
	lmp.command("group chain molecule "+str(molId))
	lmp.command("group restOfchain id "+(" ".join(atoms2del)))
	lmp.command("neigh_modify exclude group restOfchain ")
	lmp.command("delete_bonds restOfchain multi any")
	lmp.command("run 1 pre no post no")
	initial_energy = lmp.extract_compute("thermo_pe",0,0)
	
	weight0=1
	for idx in xrange(startindex,len(mol)):
		if not (mol[idx:].size==0):
			lmp.command("group restOfchain id "+(" ".join(mol[idx:])))
	                lmp.command("neigh_modify exclude group restOfchain")
		lmp.command("group beginOfchain id "+(" ".join(mol[1:idx])))
        	lmp.command("delete_bonds beginOfchain multi undo")
        	lmp.command("run 1 pre no post no")
		delta_pe = lmp.extract_compute("thermo_pe",0,0) - initial_energy
		lmp.command("neigh_modify exclude none")
		initial_energy+=delta_pe
		weight0*=exp(-beta*delta_pe)
	

if __name__ == "__main__":
	T=298
	kb = 0.0019872041
	beta = 1/(kb*T)
	tries = 200000
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
	(atoms,bonds,angles,dihedrals) = rdlmp.readAll(inputfile)
	natoms = atoms.shape[0]
	molIDs = atoms[np.where(atoms[:,2]==sulfurID)][:,1]
	ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
	meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]	
	
	(ddts,meohs) = initializeMols(atoms,bonds)
	
	lmp = lammps()
	lmp.file("ddt_me_200.lmi")
	pe = lmp.extract_compute("thermo_pe",0,0)
	
	#lmp.command("neigh_modify exclude type 1 1 delay 0 every 1 check yes")
	lmp.command("neigh_modify exclude type 1 1")
	coords = lmp.gather_atoms("x",1,3)
	atoms = atoms[atoms[:,0].argsort()]
	loop_start = time.time()		
	for i in xrange(tries):
		#iter_start = time.time()
		pe_old = pe
		atoms_old = np.copy(atoms)
		coord_old = coords
		move = rnd.choice(['swap','rotate'])
		
		if(move=='swap'):
			#print "swapping"
			ddtmol = rnd.choice(ddtMols)
			meohmol = rnd.choice(meohMols)
			swapMolecules(ddtmol,meohmol,atoms,centerRotation,rotationtype)
		elif(move=='rotate'):
			#print "rotating"
			molId = rnd.choice(molIDs)
			randomRotate(atoms,molId,max_angle)
		elif(move=='cbmc'):
			print "placeholder"	
		for idx in xrange(natoms):
			coords[idx*3]=atoms[idx,4]
			coords[idx*3+1]=atoms[idx,5]
			coords[idx*3+2]=atoms[idx,6]
		lmp.scatter_atoms("x",1,3,coords)
		lmp.command("run 1 start "+str(i)+" stop "+str(i+1)+" pre no post no")
		pe = lmp.extract_compute("thermo_pe",0,0)
		print "On loop: "+str(i)
		lmp.command("write_dump all xyz ddt_me_200.xyz modify append yes")
		if((pe<=pe_old) or (exp(-beta*(pe-pe_old))>rnd.uniform(0,1))):
			print "Move accepted"
		else:
			print "Move rejected"
			atoms = np.copy(atoms_old)
			pe = pe_old
		potentialfile.write(str(i)+'\t'+str(pe)+'\n')
		#raw_input("continue?")
		#iter_end = time.time()
		if(((i+1)%100)==0):
			iter_end = time.time()
			print "Total time is "+str(iter_end-loop_start)+" average iteration time is "+str((iter_end-loop_start)/(i+1.0))
	lmp.close()
	loop_end=time.time()-loop_start
