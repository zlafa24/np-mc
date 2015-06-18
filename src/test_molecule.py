#!/usr/bin/python
import read_lmp_rev6 as rdlmp
import numpy as np
import random as rnd
import itertools as itt

'''
def getBondedAtoms(bonds,atomID):
	bonded1 = bonds[(bonds[:,2]==atomID)][:,3] if bonds[(bonds[:,2]==atomID)].shape[0]>0 else []
	bonded2 = bonds[(bonds[:,3]==atomID)][:,2] if bonds[(bonds[:,3]==atomID)].shape[0]>0 else []
	return np.append(np.ravel(bonded1),np.ravel(bonded2))

def getMoleculeAtoms(bonds,startID):
	atomIDs = np.empty([1])
	atomIDs[0] = startID
	#print "atomID's 0: "+str(startID)
	bondedAtoms = getBondedAtoms(bonds,startID)
	actAtoms = [atom for atom in bondedAtoms if ((atom>0) and (not (atom in atomIDs)))]
	while(len(actAtoms)>0):
		#print "atomID's "+str(len(atomIDs))+": "+str(actAtoms)
		atomIDs = np.append(atomIDs,actAtoms[0])
		bondedAtoms = getBondedAtoms(bonds,actAtoms[0])
       		#print "Bonded atoms are: "+str(bondedAtoms)
		actAtoms = [atom for atom in bondedAtoms if ((atom>0) and (not (atom in atomIDs)))]
	return atomIDs
	#print str(any(((bondedAtoms>0) and (not (bondedAtoms in atomIDs)))))
	#bondedAtoms = [atom for atom in bondedAtoms if not (atom in atomIDs)]
	#print actAtoms
	
'''	

ch3ID = 3
sulfurID = 4
oxygenID = 5

inputfile = 'addmolecule_184_rand.lmp'
molecules = rdlmp.readAll(inputfile)

atoms = molecules[0]
bonds = molecules[1]

print bonds

ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]

sulfurs = atoms[np.where(atoms[:,2]==sulfurID)]
#ddtsulfurs = atoms[np.where(((atoms[:,2]==sulfurID) and (atoms[:,1] in ddtMols)))]
#meohsulfurs = atoms[np.where(((atoms[:,2]==sulfurID) and (atoms[:,1] in meohMols)))]
ddtsulfurs = [atom[0] for atom in atoms if (atom[2]==sulfurID and (atom[1] in ddtMols))]
meohsulfurs = [atom[0] for atom in atoms if (atom[2]==sulfurID and (atom[1] in meohMols))]

atomID = rnd.choice(sulfurs[:,0])

print "Sulfur atom id is: "+str(atomID)

#sbonds = bonds[np.where((bonds[:,2]==atomID) | (bonds[:,3]==atomID))]

#print getBondedAtoms(bonds,atomID)
bondAtoms = rdlmp.getMoleculeAtoms(bonds,atomID)
print bondAtoms
for atom in bondAtoms:
	print atoms[atoms[:,0]==atom][:,2]

ddts = np.empty([ddtMols.shape[0],13])
meohs = np.empty([meohMols.shape[0],5])

for idx,(ddtsulfur,meohsulfur) in enumerate(itt.izip_longest(ddtsulfurs,meohsulfurs)):
	if(not (ddtsulfur==None)):
		ddts[idx,:] = rdlmp.getMoleculeAtoms(bonds,ddtsulfur)
	if(not (meohsulfur==None)):
		meohs[idx,:] = rdlmp.getMoleculeAtoms(bonds,meohsulfur)
print ddts
print meohs
