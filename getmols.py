#!/usr/bin/python
from read_lmp_rev6 import *
import numpy as np

filename = "addmolecule.lmp"
molecules = readAll(filename)
ch3Id = 3
oId = 5
ddtmols = [int(atom[1]) for atom in molecules[0] if (atom[2]==ch3Id)]
meohmols = [int(atom[1]) for atom in molecules[0] if (atom[2]==oId)]
(ddts,meohs) = initializeMols(molecules[0],molecules[1])

print ddtmols
