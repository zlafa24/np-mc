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
import pypar

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

def quat_mult(q1,q2):
	w1,x1,y1,z1 = q1
	w2,x2,y2,z2 = q2
	w = w1*w2-x1*x2-y1*y2-z1*z2
	x = w1*x2 + x1*w2 + y1*z2 - z1*y2
	y = w1*y2 + y1*w2 + z1*x2 - x1*z2
	z = w1*z2 + z1*w2 + x1*y2 - y1*x2
	return np.array([w,x,y,z])

def rot_quat(vector,theta,rot_axis):
	rot_axis = rot_axis/np.linalg.norm(rot_axis)
	vector_mag = np.linalg.norm(vector)
	quat = np.array([cos(theta/2),sin(theta/2)*rot_axis[0],sin(theta/2)*rot_axis[1],sin(theta/2)*rot_axis[2]])
	quat_inverse = np.array([cos(theta/2),-sin(theta/2)*rot_axis[0],-sin(theta/2)*rot_axis[1],-sin(theta/2)*rot_axis[2]])
	quat = quat/np.linalg.norm(quat)
	quat_inverse = quat_inverse/(np.linalg.norm(quat_inverse)**2)
	
	vect_quat = np.array([0,vector[0],vector[1],vector[2]])/vector_mag
	new_vector = quat_mult(quat_mult(quat,vect_quat),quat_inverse)
	return new_vector[1:]*vector_mag

def rotate_dihedral_quat(dih_atoms,angle,atoms2rotate):
	rot_axis = dih_atoms[2,4:7]-dih_atoms[1,4:7]
	rot_angle = angle-calc_dih_angle(dih_atoms)
	for atom in atoms2rotate:
		atom[4:7] = rot_quat((atom[4:7]-dih_atoms[2,4:7]),rot_angle,rot_axis)+dih_atoms[2,4:7]
	return atoms2rotate



def calc_dih_angle(dih_atoms):
	b1 = dih_atoms[1,4:7]-dih_atoms[0,4:7]
	b2 = dih_atoms[2,4:7]-dih_atoms[1,4:7]
	b3 = dih_atoms[3,4:7]-dih_atoms[2,4:7]
	#b4 = np.cross(b1,b2)
	#b5 = np.cross(b2,b4)
	b2norm = b2/np.linalg.norm(b2)
	n1 = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
	n2 = np.cross(b2,b3)/np.linalg.norm(np.cross(b2,b3))
	m1 = np.cross(n1,b2norm)
	angle = atan2(np.dot(m1,n2),np.dot(n1,n2))
	angle=((angle-pi)*(-1)+2*pi)%(2*pi)
	return angle
	#angle =  atan2(np.dot(b3,b4),np.dot(b3,b5)*sqrt(np.dot(b2,b2)))
	#norm_angle = angle if angle>=0 else 2*pi+angle
	#return norm_angle

def rotate_dihedral(dih_atoms,angle,atoms2rotate):
	b2 = dih_atoms[2,4:7]-dih_atoms[1,4:7]
	b3 = dih_atoms[3,4:7]-dih_atoms[2,4:7]
	rot_axis = b2/np.linalg.norm(b2)
	init_vector = b3
	#n1 = np.cross((dih_atoms[1,4:7]-dih_atoms[0,4:7]),rot_axis)
	#n2 = np.cross((dih_atoms[3,4:7]-dih_atoms[2,4:7]),rot_axis)
	#init_angle = np.arccos(np.dot((n1/np.linalg.norm(n1)),(n2/np.linalg.norm(n2))))
	init_angle = calc_dih_angle(dih_atoms)
	rot_angle = angle-init_angle
	#print "Initial angle is "+str(init_angle)+" desired angle is "+str(angle)+" therefore rotating "+str(rot_angle)
	skewmat = np.array([[0,-rot_axis[2],rot_axis[1]],[rot_axis[2],0,-rot_axis[0]],[-rot_axis[1],rot_axis[0],0]])
	rot_matrix = np.identity(3)+sin(rot_angle)*skewmat+(2*sin(rot_angle/2)**2)*np.linalg.matrix_power(skewmat,2)
	#print "\nDihedral atoms looks like this "+str(dih_atoms[:,4:7])+"\n"
	for atom in atoms2rotate:	
		atom[4:7] = atom[4:7] - dih_atoms[2,4:7]
		atom[4:7] = np.transpose(np.dot(rot_matrix,np.transpose(atom[4:7])))+dih_atoms[2,4:7]
	#print "\nNow dihedral atoms looks like this "+str(dih_atoms[:,4:7])+"\n"
	return atoms2rotate

def update_coords(atoms,lmp):
	coords = lmp.gather_atoms("x",1,3)
	for idx in xrange(atoms.shape[0]):
		coords[idx*3]=atoms[idx,4]
		coords[idx*3+1]=atoms[idx,5]
		coords[idx*3+2]=atoms[idx,6]
	lmp.scatter_atoms("x",1,3,coords)

def delete_chain(mol,delindex,lmp,delete=True):
	if(delete):	
		atoms2del = mol[delindex:].astype(int)
		lmp.command("group restOfchain id 1")
		lmp.command("group restOfchain clear")
		lmp.command("group restOfchain id "+(" ".join([str(atom) for atom in atoms2del])))
		lmp.command("neigh_modify exclude group restOfchain all")
		lmp.command("delete_bonds restOfchain multi any")
		#lmp.command("group restOfchain delete")
	else:
		lmp.command("neigh_modify exclude none")
		lmp.command("group beginOfchain id 1")	
		lmp.command("group beginOfchain clear")
		lmp.command("group beginOfchain id "+(" ".join([str(atom) for atom in mol[0:(delindex+1)].astype(int)])))
		if(delindex<(mol.shape[0]-1)):
			lmp.command("group restOfchain id 1")
			lmp.command("group restOfchain clear")
			lmp.command("group restOfchain id "+(" ".join([str(atom) for atom in mol[(delindex+1):].astype(int)])))
			lmp.command("neigh_modify exclude group restOfchain all")
		lmp.command("delete_bonds beginOfchain multi undo")
                #lmp.command("group beginOfchain delete")
		#if(delindex<(mol.shape[0]-1)):
			#lmp.command("group restOfchain clear")

def regrow_chain(atoms,mol,beta,lmp,dih_cdf,startindex,numtrials,keep_original=False):
	totalstart = time.time()
	delete_chain(mol,startindex,lmp,delete=True)
	weight1=1
	actual_trials = numtrials-1 if keep_original else numtrials
	mol_id = int(atoms[np.where((atoms[:,0]==mol[0]))][0,1])
	lmp.command("group cbmc_mol molecule "+str(mol_id))
	lmp.command("group all_else subtract all cbmc_mol")
	lmp.command("neigh_modify exclude group all_else all_else")
	lmp.command("run 0 post no")
	energy = lmp.extract_compute("pair_pe",0,0)
	for idx in xrange(startindex,len(mol)):
		start = time.time()
		#lmp.command("neigh_modify exclude group all_else all_else")
		#lmp.command("run 0 post no")
		#energy = lmp.extract_compute("pair_pe",0,0)
		start_delchain = time.time()
		delete_chain(mol,idx,lmp,delete=False)
		finish_delchain = time.time()
		print "Undeleting chain takes "+str(finish_delchain-start_delchain)
		lmp.command("neigh_modify exclude group all_else all_else")
		lmp.command("run 0 post no")
		lmp_time = time.time()-start
		print "Time setting up neighbor list conditions is "+str(lmp_time)
		if(idx>2):
			start = time.time()
			probs = np.empty((numtrials))
			positions = np.empty((numtrials,(mol.shape[0]-idx),3))
			energies = np.empty((numtrials))
			dih_atoms = atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[(idx-3):(idx+1)]]).flatten()]
			original_pos = np.copy(atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[idx:]]).flatten(),4:7])
			chosen_pos = 0
			atoms2rotate = atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[idx:]]).flatten()]
			#print "Starting energy is "+str(energy)+"\n\n"
			finish = time.time()
			print "Initialization time is "+str(finish-start)
			if(keep_original):
				start = time.time()
				lmp.command("run 1 pre no post no")
				pe = lmp.extract_compute("pair_pe",0,0)
				delta_pe = pe-energy
				print "Angle is original, pe is "+str(pe)+" delpe is "+str(delta_pe)+"\n"
				probs[numtrials-1] = exp(-beta*delta_pe) if -beta*delta_pe<700 else float('inf')
				energies[numtrials-1] = pe
				finish = time.time()
				print "Original evaluation time is "+str(finish-start)
			#start = time.time()
			for i in xrange(actual_trials):
				start = time.time()
				chosen_index = np.searchsorted(dih_cdf[:,1],rnd.uniform(0,1))
				angle = dih_cdf[chosen_index,0]
				newpositions = rotate_dihedral_quat(dih_atoms,angle,atoms2rotate)
				positions[i,:,:] = newpositions[:,4:7]
				atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[idx:]]).flatten()] = newpositions
				#finish = time.time()
				#print "Time to update positions is "+str(finish-start)
				#dih_atoms[3,4:7] = positions[i,0,:]
				update_coords(atoms,lmp)
				#start = time.time()
				lmp.command("run 1 pre no post no")
				finish = time.time()
				print "Time to perform one energy evaluation "+str(finish-start)
				pe = lmp.extract_compute("pair_pe",0,0)
				delta_pe = pe-energy
				#print "Energy is "+str(pe)+" delPE is "+str(delta_pe)+" angle is "+str(angle)
				probs[i] = exp(-beta*(delta_pe)) if -beta*(delta_pe)<700 else float('inf')
				energies[i] = pe
			finish = time.time()
			#print "Evaluating trials takes "+str(finish-start)
			#print "Probabilities are "+str(probs)
			if(keep_original):
				energy = energies[numtrials-1]
				for count,position in enumerate(original_pos):
					atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
			else:
				angle_cdf = np.cumsum(probs)/np.sum(probs)
				chosen_pos = np.searchsorted(angle_cdf,rnd.uniform(0,1))
				energy = energies[chosen_pos]
				for count,position in enumerate(positions[chosen_pos]):
					atoms[np.where(atoms[:,0]==mol[idx+count])[0],4:7] = position
			update_coords(atoms,lmp)
			#lmp.command("run 1 pre no post no")
		stepweight=np.sum(probs) if idx>2 else 1
		weight1*=stepweight
	lmp.command("neigh_modify exclude none")
	lmp.command("neigh_modify exclude type 1 1")
	totalfinish = time.time()
	print "Total regrow time is "+str(totalfinish-totalstart)
	return (weight1,atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[startindex:]]).flatten()])
			


def cbmc(atoms,mol,beta,lmp,dih_cdf,startindex):
	molId = atoms[atoms[:,0]==mol[0]][0,1]
	#print "Mol id is "+str(molId)
	numtrials = 5
	#startindex = rnd.choice(range(1,len(mol)))
	atoms2del = mol[startindex:].astype(int)
	#print "Atoms to delete are "+str(atoms2del)
	(weight0,original_positions) = regrow_chain(atoms,mol,beta,lmp,dih_cdf,startindex,numtrials,keep_original=True)
	(weight1,new_positions) = regrow_chain(atoms,mol,beta,lmp,dih_cdf,startindex,numtrials)
	acceptance_prob = weight1/weight0 if weight0>0 else float('inf')
	print "Weight1 is "+str(weight1)+" weight0 is "+str(weight0)
	if(rnd.uniform(0,1)>acceptance_prob):
		print atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[startindex:]]).flatten()].shape
		print original_positions.shape
		atoms[np.array([np.where(atoms[:,0]==atom)[0] for atom in mol[startindex:]]).flatten()] = original_positions
		update_coords(atoms,lmp)
		return (False,weight1,weight0,original_positions)
	else:
		return (True,weight1,weight0,original_positions)

def randomShift(atoms,molId,max_radius):
	atomIndices = np.where((atoms[:,1]==molId))[0]
	radius = rnd.uniform(0,max_radius)
	phi = rnd.uniform(0,2*pi)
	theta = rnd.uniform(0,pi)
	shiftX = radius*sin(theta)*cos(phi)
	shiftY = radius*sin(theta)*sin(phi)
	shiftZ = radius*cos(theta)	
	shiftMolecule(atoms,atomIndices,shiftX,shiftY,shiftZ)

if __name__ == "__main__":
	T=278
	kb = 0.0019872041
	beta = 1/(kb*T)
	tries = 400000
	agID = 1
	ch2ID = 2
	ch3ID = 3
	sulfurID = 4
	oxygenID = 5
	hydrogenID = 6

	centerRotation = [40.9,40.9,40.9]
	rotationtype = 'align'

	potentialfile = open('Potential_Energy.txt','w')
	potentialfile.write('Step\tPotential (kcal/mol)\tMove\n')

	acceptancefile = open('Acceptance_Rates.txt','w')
	acceptancefile.write('Step\t#Swaps\tRate\t#Rotations\tRate\t#CBMC\tRate\n')

	dih_potential = np.loadtxt("dih_potential_1",skiprows=1)
	dih_cdf = np.cumsum(np.exp(-beta*dih_potential[:,1]))
	dih_cdf_norm = dih_cdf/dih_cdf[dih_cdf.shape[0]-1]
	dih_cdf_norm = np.hstack((dih_potential[:,0].reshape((dih_cdf.shape[0],1)),dih_cdf_norm.reshape((dih_cdf.shape[0],1))))
	
	max_angle = 0.34906585
	max_radius = 1.4	

	inputfile = 'addmolecule_184_rand.lmp'
	(atoms,bonds,angles,dihedrals) = rdlmp.readAll(inputfile)
	print "Initializing atoms from lmp input file..."
    natoms = atoms.shape[0]
	molIDs = atoms[np.where(atoms[:,2]==sulfurID)][:,1]
	ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
	meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]
    print "Finding molecules..."
	(ddts,meohs) = initializeMols(atoms,bonds)
    print "Starting up lammps instance..."
	lmp = lammps("",["-echo","none","-screen","lammps.out"])
	#lmp = lammps("",["-echo","none"])
    print "Running equilibration steps..."
	lmp.file("ddt_me_200.lmi")
	lmp.command("compute pair_pe all pe pair")
	lmp.command("neigh_modify exclude type 1 1")
	#lmp.command("compute unique all pe/atom pair")
	lmp.command("run 1 pre no post yes")
	#cont = raw_input("continue?")
	pe = lmp.extract_compute("thermo_pe",0,0)
	#print "Extracted pe is "+str(pe)
	#print "About to print per atom energies"
	#print "Per atom energies are: "
	#pe_atom1 = lmp.extract_compute("unique",1,1)[0]
	#print pe_atom1
	#forces = lmp.extract_atom("type",0)
	#for i in xrange(4299):
	#	print forces[i]
	coords = lmp.gather_atoms("x",1,3)
	atoms = atoms[atoms[:,0].argsort()]
	loop_start = time.time()
	swaps=0
	swaps_accepted=0
	rotates=0
	rotates_accepted=0
	shifts = 0
	shifts_accepted = 0
	cbmcs=0
	cbmcs_accepted=0
	for i in xrange(tries):
		#iter_start = time.time()
		potentialfile.flush()
		pe_old = pe
		atoms_old = np.copy(atoms)
		coord_old = coords
		move = rnd.choice(['swap','cbmc','rotate','shift'])
		if((i+1)%100==0):
			acceptancefile.write(str(i)+'\t'+str(swaps)+'\t'+str(float(swaps_accepted)/float(swaps+0.01))+'\t'+str(rotates)+'\t'+str(float(rotates_accepted)/float(rotates+0.01))+'\t'+str(cbmcs)+'\t'+str(float(cbmcs_accepted)/float(cbmcs+0.01))+'\t'+str(shifts)+'\t'+str(float(shifts_accepted)/float(shifts+0.01))+'\n')
			lmp.command("write_data configuration_"+str(i)+".lmp")
			acceptancefile.flush()
			swaps=0
			swaps_accepted=0
			rotates=0
			rotates_accepted=0
			shifts=0
			shifts_accepted=0
			cbmcs=0
			cbmcs_accepted=0
		if(move=='swap'):
			start = time.time()
			swaps+=1
			print "swapping"
			ddtmol = rnd.choice(ddts)
			meohmol = rnd.choice(meohs)
			ddtmol_id = atoms[(atoms[:,0]==ddtmol[0])][:,1]
			meohmol_id = atoms[(atoms[:,0]==meohmol[0])][:,1]
			#ddtmol_id = rnd.choice(ddtMols)
			#meohmol_id = rnd.choice(meohMols)
			swapMolecules(ddtmol_id,meohmol_id,atoms,centerRotation,rotationtype)
			#ddt_accepted = cbmc(atoms,ddtmol,beta,lmp,dih_cdf_norm,3)
			#meoh_accepted = cbmc(atoms,meohmol,beta,lmp,dih_cdf_norm,3)
			finish = time.time()
			print "Swap time is "+str(finish-start)
		elif(move=='rotate'):
			rotates+=1
			print "rotating"
			molId = rnd.choice(molIDs)
			randomRotate(atoms,molId,max_angle)
		elif(move=='shift'):
			shifts+=1
			print "shifting"
			molId = rnd.choice(molIDs)
			randomShift(atoms,molId,max_radius)
		elif(move=='cbmc'):
			cbmcs+=1
			print "\n\nCBMC Move\n\n"
			mols = rnd.choice((ddts,meohs))
			mol = rnd.choice(mols)
			startindex = rnd.choice(np.arange(1,len(mol)))
			accepted = cbmc(atoms,mol,beta,lmp,dih_cdf_norm,startindex)
			lmp.command("write_dump all xyz ddt_me_200.xyz modify append yes")
			if(accepted[0]):
				cbmcs_accepted+=1
				print "Move accepted"
				lmp.command("run 1 post no")
				pe = lmp.extract_compute("thermo_pe",0,0)
				print "\nOn loop "+str(i)+"\n"
				potentialfile.write(str(i)+'\t'+str(pe)+'\t'+move+' succeeded W1:'+str(accepted[1])+' W0:'+str(accepted[2])+'\n')
				continue
			else:
				print "Move rejected"
				pe = pe_old
				atoms = atoms_old
				print "\nOn loop "+str(i)+"\n"
				potentialfile.write(str(i)+'\t'+str(pe)+'\t'+move+' failed W1:'+str(accepted[1])+' W0:'+str(accepted[2])+'\n')
				continue
		update_coords(atoms,lmp)
		lmp.command("run 1 post no")
		pe = lmp.extract_compute("thermo_pe",0,0)
		print "On loop: "+str(i)
		lmp.command("write_dump all xyz ddt_me_200.xyz modify append yes")
		#print "per atom energies: "+str(lmp.extract_compute("peratom",1,3))
		print "New PE is: "+str(pe)+" Old PE is: "+str(pe_old)+"Delta PE is "+str(pe-pe_old)
		if((pe<=pe_old) or (exp(-beta*(pe-pe_old))>rnd.uniform(0,1))):
			print "Move accepted"
			if(move=='rotate'):
				rotates_accepted+=1
			elif(move=='swap'):
				swaps_accepted+=1
			elif(move=='shift'):
				shifts_accepted+=1
		else:
			print "Move rejected"
			atoms = atoms_old
			pe = pe_old
		potentialfile.write(str(i)+'\t'+str(pe)+'\t'+move+'\n')
		if(((i+1)%100)==0):
			iter_end = time.time()
			print "Total time is "+str(iter_end-loop_start)+" average iteration time is "+str((iter_end-loop_start)/(i+1.0))
	lmp.close()
	loop_end=time.time()-loop_start
