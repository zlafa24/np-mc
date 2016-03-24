#!/home/snm8xf/anaconda/bin/python
import read_lmp_rev6 as rdlmp
import sys
import numpy as np
import itertools as itt

def initializeMols(atoms,bonds):
    ch3ID = 3
    sulfurID = 4
    oxygenID = 5
    
    ddtMols = atoms[np.where(atoms[:,2]==ch3ID)][:,1]
    meohMols = atoms[np.where(atoms[:,2]==oxygenID)][:,1]

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

if __name__=="__main__":
    setcharge=0.153
    filename=sys.argv[1]
    molecules = rdlmp.readAll(filename)
    atoms=molecules[0]
    print atoms
    (ddts,meohs) = initializeMols(molecules[0],molecules[1])
    for atom in ddts[:,3]:
        print "Adding charge to atom #"+str(atom)
        print np.where(atoms[:,0]==int(atom))
        atoms[np.where(atoms[:,0]==int(atom))[0],3]=setcharge
    rdlmp.editFile(filename,"charged.lmp",atoms,molecules[1],molecules[2],molecules[3],) 
    #print str(ddts)
    #print str(meohs)
