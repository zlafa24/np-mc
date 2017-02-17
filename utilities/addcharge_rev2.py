#!/usr/bin/python
import read_lmp_rev6 as rdlmp
import sys
import numpy as np
import itertools as itt

if __name__=="__main__":
	ch2_charge=0.265
	o_charge = -0.700
	h_charge = 0.435
	filename=sys.argv[1]
	print "Reading file"
	molecules = rdlmp.readAll(filename)
	atoms=molecules[0]
	print atoms
	(ddts,meohs,eths) = rdlmp.initializeMols(molecules[0],molecules[1])
	print "MeOH append job "+str(np.append(meohs[:,2],eths[:,1]))
	for atom1,atom2,atom3 in zip(np.append(meohs[:,2],eths[:,1]),np.append(meohs[:,3],eths[:,2]),np.append(meohs[:,4],eths[:,3])):
		atoms[np.where(atoms[:,0]==int(atom1))[0],3]=ch2_charge
		atoms[np.where(atoms[:,0]==int(atom2))[0],3]=o_charge
		atoms[np.where(atoms[:,0]==int(atom3))[0],3]=h_charge
	rdlmp.editFile(filename,"charged.lmp",atoms,molecules[1],molecules[2],molecules[3],) 
