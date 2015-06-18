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

def cbmc(atoms,mol,lmpptr):
	lmp = lammps(ptr=lmptr)
	startindex = rnd.choice(range(1,len(mol)))
	atoms2del = mol[startindex:]
	lmp.command("group restOfchain id "+(" ".join(atoms2del)))
	lmp.command("neigh_modify exclude group restOfchain")
	lmp.command("delete_bonds restOfchain multi any")
	lmp.command("run 0 pre no post no")
	for idx in xrange(startindex,len(mol)):
		print "placeholder"


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
	#pe = getPotential('pe.out')
	(atoms,bonds,angles,dihedrals) = rdlmp.readAll(inputfile)
	natoms = atoms.shape[0]
	molIDs = atoms[np.where(atoms[:,2]==sulfurID)][:,1]
	ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
	meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]	
	#coords = POINTER(POINTER(c_double*3)*natoms)()
	
	(ddts,meohs) = initializeMols(atoms,bonds)
	
	lmp = lammps()
	#lmp.file('addmolecule_184_rand.lmp')
	lmp.file("ddt_me_200.lmi")
	pe = lmp.extract_compute("thermo_pe",0,0)
	lmp.command("neigh_modify delay 0 every 1 check yes")
	#ids = lmp.extract_atom("id",0)
	#mol_ids = lmp.extract_atom("mol",0)
	#types = lmp.extract_atom("type",0)
	#xtrct_coords = lmp.extract_atom("x",3)
	#for idnumber in xrange(natoms):
		#print "Atom "+str(ids[idnumber])+" x: "+str(xtrct_coords[idnumber][0])+" y: "+str(xtrct_coords[idnumber][1])+" z: "+str(xtrct_coords[idnumber][2])
		#print mol_ids
		#print mol_ids[idnumber]
	#	atoms[idnumber,0] = ids[idnumber]
	#	atoms[idnumber,1] = atoms[atoms[:,0]==ids[idnumber]][0,1]
	#	atoms[idnumber,2] = atoms[atoms[:,0]==ids[idnumber]][0,2]
	#	atoms[idnumber,3] = 0
	#	atoms[idnumber,4] = xtrct_coords[idnumber][0]
	#	atoms[idnumber,5] = xtrct_coords[idnumber][1]
	#	atoms[idnumber,6] = xtrct_coords[idnumber][2]
	coords = lmp.gather_atoms("x",1,3)
	atoms = atoms[atoms[:,0].argsort()]
		
	for i in xrange(tries):
		pe_old = pe
		atoms_old = np.copy(atoms)
		coord_old = coords
		move = rnd.choice(['swap','rotate'])
		mismatched = 0
		#for coord in xrange(natoms):
			#print "Coords "+str(coord+1)+" are  x: "+str(coords[coord*3])+" y: "+str(coords[coord*3+1])+" z: "+str(coords[coord*3+2])
			#print "Atom# is "+str(atoms[coord,0])
		#	if(not (coords[coord*3]==atoms[coord,4])):
		#		mismatched+=1
		#	if(not (coords[coord*3+1]==atoms[coord,5])):
                #                mismatched+=1
		#	if(not (coords[coord*3+2]==atoms[coord,6])):
                #                mismatched+=1	
		if(move=='swap'):
			print "swapping"
			ddtmol = rnd.choice(ddtMols)
			meohmol = rnd.choice(meohMols)
			swapMolecules(ddtmol,meohmol,atoms,centerRotation,rotationtype)
		elif(move=='rotate'):
			print "rotating"
			molId = rnd.choice(molIDs)
			randomRotate(atoms,molId,max_angle)
		#print "x range is: "+str(np.amin(atoms[:,4]))+"-"+str(np.amax(atoms[:,4]))
		#print "y range is: "+str(np.amin(atoms[:,5]))+"-"+str(np.amax(atoms[:,5]))
		#print "z range is: "+str(np.amin(atoms[:,6]))+"-"+str(np.amax(atoms[:,6]))
		#print "Number mismatched is "+str(mismatched)
		for idx in xrange(natoms):
			coords[idx*3]=atoms[idx,4]
			coords[idx*3+1]=atoms[idx,5]
			coords[idx*3+2]=atoms[idx,6]
		#raw_input("continue?")
		lmp.scatter_atoms("x",1,3,coords)
		lmp.command("run 1 pre no post no")
		pe = lmp.extract_compute("thermo_pe",0,0)
		print "On loop: "+str(i)
		print "New pe is "+str(pe)+" old pe is "+str(pe_old)+" deltPE is "+str(pe-pe_old)
		lmp.command("write_dump all xyz ddt_me_200.xyz modify append yes")
		if((pe<=pe_old) or (exp(-beta*(pe-pe_old))>rnd.uniform(0,1))):
			print "Move accepted"
		else:
			print "Move rejected"
			atoms = np.copy(atoms_old)
			pe = pe_old
		potentialfile.write(str(i)+'\t'+str(pe)+'\n')
		#raw_input("continue?")
	lmp.close()
