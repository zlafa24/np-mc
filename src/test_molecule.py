#!/usr/bin/python
import read_lmp_rev6 as rdlmp
import numpy as np
import random as rnd
import itertools as itt

if __name__ == "__main__":
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
