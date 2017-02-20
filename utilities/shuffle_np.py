#!/usr/bin/python
import numpy as np
import random as rnd
import mc_library_rev3 as mcl
import read_lmp_rev6 as rdlmp
import sys

if __name__ == "__main__":
    sulfurType = 4
    ch3Type = 3
    oType = 5

    filename = sys.argv[1]
    molecules = rdlmp.readAll(filename)
    atoms = molecules[0]

    ddts = atoms[np.where(atoms[:,2]==ch3Type)][:,1]
    meohs = atoms[np.where(atoms[:,2]==oType)][:,1]

    for i in xrange(5000):
            if((i+1)%100):
                    print "Surface swapped "+str(i)+" times"
            mcl.swapMolecules(rnd.choice(ddts),rnd.choice(meohs),atoms,[61.4803,61.27595,61.4293],"align")

    xyzfile = open("addmolecule.xyz",'w')
    xyzfile.write(str(atoms.shape[0])+"\n\n")
    np.savetxt("temp.xyz",atoms[:,[2,4,5,6]],'%d %f %f %f')
    atomtmp = open("temp.xyz",'r')
    xyzfile.write(atomtmp.read())

    rdlmp.editFile(filename,"lammps_shuffled.lmp",molecules[0],molecules[1],molecules[2],molecules[3])
