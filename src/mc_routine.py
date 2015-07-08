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
from scipy.stats import rv_discrete


def atom2xyz(filename,atoms):
	xyzfile = open(filename,'a')
	xyzfile.write(str(atoms.shape[0])+'\n\n')
	for atom in atoms:
		xyzfile.write(str(atom[2])+'\t'+str(atom[4])+'\t'+str(atom[5])+'\t'+str(atom[6])+'\n')

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


def rotate_dihedral(dih_atoms,angle,atoms2rotate):
	rot_axis = (dih_atoms[2,4:7]-dih_atoms[1,4:7])/np.linalg.norm((dih_atoms[2,4:7]-dih_atoms[1,4:7]))
	init_vector = dih_atoms[3,4:7]-dih_atoms[2,4:7]
	skewmat = np.array([[0,-rot_axis[2],rot_axis[1]],[rot_axis[2],0,-rot_axis[0]],[-rot_axis[1],rot_axis[0],0]])
	rot_matrix = np.identity(3)+sin(angle)*skewmat+(2*sin(angle)**2)*np.linalg.matrix_power(skewmat,2)
	for atom in atoms2rotate:	
		atom[4:7] = atom[4:7] - dih_atoms[2,4:7]
		atom[4:7] = np.transpose(np.dot(rot_matrix,np.transpose(atom[4:7])))+dih_atoms[2,4:7]
	return atoms2rotate

def update_coords(atoms,lmp,mol,idx):
	coords = lmp.gather_atoms("x",1,3)
	#dih_atoms = atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[(idx-3):(idx+1)]]).flatten()]
	#print "Dihedrals inside update function are "+str(dih_atoms)
	for idx in xrange(atoms.shape[0]):
		coords[idx*3]=atoms[idx,4]
		coords[idx*3+1]=atoms[idx,5]
		coords[idx*3+2]=atoms[idx,6]
	lmp.scatter_atoms("x",1,3,coords)

def delete_chain(mol,delindex,lmp,delete=True):
	if(delete):	
		atoms2del = mol[delindex:].astype(int)
		lmp.command("group restOfchain id "+(" ".join([str(atom) for atom in atoms2del])))
		lmp.command("neigh_modify exclude group restOfchain all")
		lmp.command("delete_bonds restOfchain multi any")
		lmp.command("group restOfchain delete")
	else:
		lmp.command("neigh_modify exclude none")
		lmp.command("group beginOfchain id "+(" ".join([str(atom) for atom in mol[0:(delindex+1)].astype(int)])))
		if(delindex<(mol.shape[0]-1)):	
			lmp.command("group restOfchain id "+(" ".join([str(atom) for atom in mol[(delindex+1):].astype(int)])))
			lmp.command("neigh_modify exclude group restOfchain all")
		lmp.command("delete_bonds beginOfchain multi undo")
                lmp.command("group beginOfchain delete")
		if(delindex<(mol.shape[0]-1)):
			lmp.command("group restOfchain delete")


def cbmc(atoms,mol,beta,lmp,dih_cdf):
	molId = atoms[atoms[:,0]==mol[0]][0,1]
	#print "Mol id is "+str(molId)
	numtrials = 5
	startindex = rnd.choice(range(1,len(mol)))
	atoms2del = mol[startindex:].astype(int)
	#print "Atoms to delete are "+str(atoms2del)
	delete_chain(mol,startindex,lmp,delete=True)
	lmp.command("run 1 post no")
	initial_energy = lmp.extract_compute("thermo_pe",0,0)
	energy = initial_energy
	weight0=1
	for idx in xrange(startindex,len(mol)):
		delete_chain(mol,idx,lmp,delete=False)
        	lmp.command("run 1 post no")
		delta_pe = lmp.extract_compute("thermo_pe",0,0) - energy
		#print "Change in energy is "+str(delta_pe)
		energy+=delta_pe
		weight0*=exp(-beta*delta_pe)
	#print "Initial rosenbluth weight is "+str(weight0)
	delete_chain(mol,startindex,lmp,delete=True)
	lmp.command("run 1 post no")
	energytrial = initial_energy
	weight1=1
	for idx in xrange(startindex,len(mol)):
		delete_chain(mol,idx,lmp,delete=False)
		energy = lmp.extract_compute("thermo_pe",0,0)
		if(idx>2):
			probs = np.empty((numtrials))
			positions = np.empty((numtrials,(mol.shape[0]-idx),3))
			dih_atoms = atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[(idx-3):(idx+1)]]).flatten()]
			#print "Dihedral atoms are "+str(mol[(idx-3):(idx+1)])
			#print "Old positions are "+str(dih_atoms[:,4:7])
			original_pos = atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[idx:]]).flatten(),4:7]
			chosen_pos = 0
			select=False
			energytrial = lmp.extract_compute("thermo_pe",0,0)
			atoms2rotate = atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[idx:]]).flatten()]
			for i in xrange(numtrials):
				chosen_index = np.searchsorted(dih_cdf[:,1],rnd.uniform(0,1))
				angle = dih_cdf[chosen_index,0]
				#new_pos = rotate_dihedral(dih_atoms,angle)
				positions[i,:,:] = rotate_dihedral(dih_atoms,angle,atoms2rotate)[:,4:7]
				#print "Positions are"+str(positions[i])
				#for count,atom in enumerate(mol[idx:]):
				#	positions[i,count,:] = np.dot(rot_matrix,(atoms[np.where(atoms[:,0]==atom)[0],4:7]-dih_atoms[2,4:7]))+dih_atoms[2,4:7]
				#positions[i,:]  = new_pos
				for count,position in enumerate(positions[i]):
					atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
				#print "Atoms are "+str(atoms)
				#print "New positions are "+str(positions[i,:,:])
				update_coords(atoms,lmp,mol,idx)
				lmp.command("run 1 post no")
				probs[i] = exp(-beta*(lmp.extract_compute("thermo_pe",0,0)-energytrial)) if -beta*(lmp.extract_compute("thermo_pe",0,0)-energytrial)<700 else float('inf')
				if(probs[i]==float('inf')):
					select=True
					chosen_pos =i
                                 	break
				for count,position in enumerate(original_pos):
                                        atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
			if(select):
				for count,position in enumerate(positions[chosen_pos]):
                                        atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
			elif(np.sum(probs)>0):
				angle_cdf = rv_discrete(values = (np.arange(numtrials),probs/np.sum(probs)))
				chosen_pos = angle_cdf.rvs(size=1)
				#atoms[np.where(atoms[:,0]==mol[idx])[0],4:7] = positions[chosen_pos,:]
				#print "Positions are "+str(positions)
				#print "Chosen position is "+str(positions[chosen_pos][0])
				for count,position in enumerate(positions[chosen_pos][0]):
                                        #print "Shape of position is "+str(position.shape)
					#print "Shape of atoms is "+str(atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7].shape)
					atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
			else:
				#atoms[np.where(atoms[:,0]==mol[idx])[0],4:7] = original_pos
				for count,position in enumerate(original_pos):
                                        atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
                       #print "New positions are "+str(positions)
			update_coords(atoms,lmp,mol,idx)
			lmp.command("run 1 post no")
		delta_pe = lmp.extract_compute("thermo_pe",0,0) - energy
		stepweight=exp(-beta*delta_pe) if -beta*delta_pe<700 else float('inf')
		weight1*=stepweight
	if(rnd.uniform(0,1)>(weight1/weight0)):
		return False
	else:
		return True

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

	centerRotation = [40.9,40.9,40.9]
	rotationtype = 'swap'

	potentialfile = open('Potential_Energy.txt','w')
	potentialfile.write('Step\tPotential (kcal/mol)\n')

	dih_potential = np.loadtxt("dih_potential_1",skiprows=1)
	dih_cdf = np.cumsum(np.exp(-beta*dih_potential[:,1]))
	dih_cdf_norm = dih_cdf/dih_cdf[dih_cdf.shape[0]-1]
	dih_cdf_norm = np.hstack((dih_potential[:,0].reshape((dih_cdf.shape[0],1)),dih_cdf_norm.reshape((dih_cdf.shape[0],1))))
	max_angle = 0.34906585
	inputfile = 'addmolecule_184_rand.lmp'
	(atoms,bonds,angles,dihedrals) = rdlmp.readAll(inputfile)
	natoms = atoms.shape[0]
	molIDs = atoms[np.where(atoms[:,2]==sulfurID)][:,1]
	ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
	meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]

	(ddts,meohs) = initializeMols(atoms,bonds)

	lmp = lammps("",["-echo","none","-screen","lammps.out"])
	lmp.file("ddt_me_200.lmi")
	pe = lmp.extract_compute("thermo_pe",0,0)

	lmp.command("neigh_modify exclude type 1 1")
	coords = lmp.gather_atoms("x",1,3)
	atoms = atoms[atoms[:,0].argsort()]
	loop_start = time.time()
	for i in xrange(tries):
		#iter_start = time.time()
		pe_old = pe
		atoms_old = np.copy(atoms)
		coord_old = coords
		move = rnd.choice(['swap','rotate','cbmc'])

		if(move=='swap'):
			print "swapping"
			ddtmol = rnd.choice(ddtMols)
			meohmol = rnd.choice(meohMols)
			swapMolecules(ddtmol,meohmol,atoms,centerRotation,rotationtype)
		elif(move=='rotate'):
			print "rotating"
			molId = rnd.choice(molIDs)
			randomRotate(atoms,molId,max_angle)
		elif(move=='cbmc'):
			print "\n\nCBMC Move\n\n"
			mols = rnd.choice((ddts,meohs))
			mol = rnd.choice(mols)
			accepted = cbmc(atoms,mol,beta,lmp,dih_cdf_norm)
			if(accepted):
				print "Move accepted"
				continue
			else:
				print "Move rejected"
				atoms = atoms_old
				continue
		for idx in xrange(natoms):
			coords[idx*3]=atoms[idx,4]
			coords[idx*3+1]=atoms[idx,5]
			coords[idx*3+2]=atoms[idx,6]
		lmp.scatter_atoms("x",1,3,coords)
		lmp.command("run 1 pre no post no")
		pe = lmp.extract_compute("thermo_pe",0,0)
		print "On loop: "+str(i)
		lmp.command("write_dump all xyz ddt_me_200.xyz modify append yes")
		print "Delta PE is "+str(pe-pe_old)
		if((pe<=pe_old) or (exp(-beta*(pe-pe_old))>rnd.uniform(0,1))):
			print "Move accepted"
		else:
			print "Move rejected"
			atoms = atoms_old
			pe = pe_old
		potentialfile.write(str(i)+'\t'+str(pe)+'\n')
		#raw_input("continue?")
		#iter_end = time.time()
		if(((i+1)%100)==0):
			iter_end = time.time()
			print "Total time is "+str(iter_end-loop_start)+" average iteration time is "+str((iter_end-loop_start)/(i+1.0))
	lmp.close()
	loop_end=time.time()-loop_start
