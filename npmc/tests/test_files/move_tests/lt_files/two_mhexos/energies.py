import numpy as np
from scipy.spatial import distance as dist

def LJ(atoms,LJ_params,allowed_pairs,distances):
    energy = 0
    for i,row in enumerate(distances):
        for j,r in enumerate(row):
            if j < i: continue
            if [i+1,j+1] not in allowed_pairs: continue
            atom1_type = int(atoms[i,0]); atom2_type = int(atoms[j,0])
            if atom1_type > atom2_type: atom1_type,atom2_type = atom2_type,atom1_type
            s = LJ_params[(atom1_type,atom2_type)][1]
            E = LJ_params[(atom1_type,atom2_type)][0]
            pair_nrg = 4*E*((s/r)**12-(s/r)**6)
            print(atom1_type,atom2_type,E,s)
            energy += pair_nrg  
    return energy

def Coulombic(atoms,LJ_params,allowed_pairs,distances):
    energy = 0
    for i,row in enumerate(distances):
        for j,r in enumerate(row):
            if j < i: continue
            if [i+1,j+1] not in allowed_pairs: continue
            atom1_charge = atoms[i,1]; atom2_charge = atoms[j,1]
            pair_nrg = 332.0636*((atom1_charge*atom2_charge)/r)*np.exp(-0.2*r)
            energy += pair_nrg      
    return energy

atoms = np.genfromtxt('system.data',skip_header=30,max_rows=20,usecols=(np.arange(2,7)))
LJ_params = {}
LJ_params_array = np.genfromtxt('test.in.settings',max_rows=36,usecols=(1,2,4,5))
for row in LJ_params_array:
    LJ_params[tuple(row[0:2].astype(int))] = list(row[2:4])
dihedrals = np.genfromtxt('system.data',skip_header=94,usecols=(np.arange(2,6)),dtype=int)
distances = dist.squareform(dist.pdist(atoms[:,2:],metric='euclidean'))

allowed_pairs = []
for i,atom1 in enumerate(atoms):
    for j,atom2 in enumerate(atoms):
        if i != j:
            allowed = True
            for dihedral in dihedrals:
                if i+1 in dihedral and j+1 in dihedral: allowed = False
            #if i+1==9 or j+1==9: allowed = False
            if i+1 not in list(np.arange(1,10)): allowed = False
            if allowed: allowed_pairs.append([i+1,j+1])

LJ_energy = LJ(atoms,LJ_params,allowed_pairs,distances)
C_energy = Coulombic(atoms,LJ_params,allowed_pairs,distances)
print(LJ_energy,C_energy)