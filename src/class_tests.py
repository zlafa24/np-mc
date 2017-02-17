#!/usr/bin/python
import atom_class as atm
import molecule_class as mol
from itertools import groupby,combinations, permutations
import networkx as ntwkx
import matplotlib.pyplot as plt

datafile = '/home/slow89/np-mc/lt_files/nanoparticle/system.data'

#atoms = atm.loadAtoms(datafile)
#bonds = mol.loadBonds(datafile)
#angles = mol.loadAngles(datafile)
#dihedrals = mol.loadDihedrals(datafile)

#print len(dihedrals)

#molecules = {}

#for k,g in groupby(atoms,key=(lambda x: x.get_mol_ID())):
#    molecules[k]=list(g)


molecules = mol.constructMolecules(datafile)

test_molecule = molecules[2500]

mol_graph = mol.molecule2graph(test_molecule.atoms,test_molecule.bonds)

print ntwkx.degree(mol_graph)

#print molecules
