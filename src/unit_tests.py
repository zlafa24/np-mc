#!/home/snm8xf/anaconda/bin/python
from math import *
import read_lmp_rev6 as rdlmp
import mc_routine as mcr
import numpy as np
import matplotlib.pyplot as plt
from lammps import lammps
from subprocess import call
import os
import random as rnd

def evaluate_new_coords(lmp,atoms):
    new_atoms = np.copy(atoms)
    lmp.command("run 0 pre yes post no")
    old_energy = get_total_pe(lmp)
    get_coords(lmp,atoms)
    old_atoms = np.copy(atoms)
    mcr.update_coords(new_atoms,lmp)
    lmp.command("run 0 pre yes post no")
    new_energy = get_total_pe(lmp)
    mcr.update_coords(old_atoms,lmp)
    atoms = new_atoms
    return (old_energy,new_energy)

def print_atoms(gatoms,natoms):
    for i in range(natoms):
        print str(gatoms[i*3])+" "+str(gatoms[i*3+1])+" "+str(gatoms[i*3+2])+"\n"

def get_coords(lmp,atoms):
    indxs = np.argsort(atoms[:,0],axis=0)
    coords = lmp.gather_atoms("x",1,3)
    for idx,i in zip(indxs,xrange(atoms.shape[0])):
        atoms[idx,4]=float(coords[i*3])
        atoms[idx,5]=float(coords[i*3+1])
        atoms[idx,6]=float(coords[i*3+2])

def make_dih_cdf(filename,beta):
    dih_potential = np.loadtxt(filename,skiprows=1)
    dih_cdf = np.cumsum(np.exp(-beta*dih_potential[:,1]))
    dih_cdf_norm = dih_cdf/dih_cdf[dih_cdf.shape[0]-1]
    dih_cdf_norm = np.hstack((dih_potential[:,0].reshape((dih_cdf.shape[0],1)),dih_cdf_norm.reshape((dih_cdf.shape[0],1))))
    return dih_cdf_norm 

def test_make_dih_cdf():
    kb = 0.0019872041
    T = 298.15
    beta = 1.0/(kb*T)
    dih_cdf = make_dih_cdf("/scratch/snm8xf/np-mc/inputfiles/dih_potential_1",beta)
    plt.plot(dih_cdf[:,0],dih_cdf[:,1])
    plt.show()

def turn_on_atoms(lmp,atomIDS,orig_charges):
    atomlist = ' '.join(map(str,map(int,atomIDS)))
    lmp.command("group onatoms id "+atomlist)
    lmp.command("group newoffatoms subtract offatoms onatoms")
    lmp.command("group offatoms clear")
    lmp.command("group offatoms union newoffatoms")
    lmp.command("neigh_modify exclude none")
    lmp.command("neigh_modify exclude group offatoms all")
    for atom in atomIDS:
        charge = orig_charges[orig_charges[:,0]==atom][:,1]
        print "Setting charge to "+str(charge[0])
        lmp.command("set atom "+str(int(atom))+" charge "+str(charge[0]))

def turn_off_atoms(lmp,atomIDS):
    stratoms = ' '.join(map(str,map(int,atomIDS)))
    print "group offatoms id "+stratoms
    lmp.command("group offatoms id "+stratoms)
    lmp.command("set group offatoms charge 0.00")
    lmp.command("neigh_modify exclude group offatoms all")

def dump_atoms(lmp,filename):
    lmp.command("write_dump all xyz "+filename+" modify append yes")

def start_lammps(config_file):
    lmp = lammps("",["-echo","none","-screen","lammps.out"])
    dname = os.path.dirname(config_file)
    print 'Directory name is '+dname
    #call(["cd",dname],shell=True)
    os.chdir(dname)
    lmp.file(config_file)
    #lmp.command("dump xyzdump all xyz 1 regrow.xyz")
    #lmp.command("dump_modify xyzdump element Ag C C S O H")
    lmp.command("compute pair_pe all pe pair")
    lmp.command("compute mol_pe all pe dihedral")
    lmp.command("fix pe_out all ave/time 1 1 1 c_thermo_pe c_pair_pe file pe.out") 
    lmp.command("thermo 1")
    lmp.command("minimize 1e-3 1e-5 200 2000")
    return lmp 

def calc_beta(T):
    kb = 0.0019872041
    T = 298.15
    return 1.0/(kb*T)

def get_total_pe(lmp):
    return lmp.extract_compute("thermo_pe",0,0)

def get_pair_pe(lmp):
    return lmp.extract_compute("pair_pe" ,0,0)

def get_mol_pe(lmp):
    return lmp.extract_compute("mol_pe",0,0)

def get_atom(atoms,atomID):
    return atoms[atoms[:,0]==int(atomID)][0]

def rot_dih_index(atoms,mol,angle,index):
    if index<3:
        return
    rotIDS = mol[index:]
    dihIDS = mol[(index-3):(index+1)]
    dih_atoms = np.array([get_atom(atoms,dihIDS[0]),get_atom(atoms,dihIDS[1]),get_atom(atoms,dihIDS[2]),get_atom(atoms,dihIDS[3])])
    atoms2rotate = np.array([atom for atom in atoms if (atom[0] in rotIDS)])
    atoms2rotate = mcr.rotate_dihedral_quat(dih_atoms,angle,atoms2rotate)
    for atom in atoms2rotate:
        atoms[atoms[:,0]==atom[0]]=atom
    mcr.update_coords(atoms,lmp)

def get_rot_trials(lmp,atoms,mol,dih_cdf,index,ntrials,keep_orig):
    pair_pe = np.zeros(ntrials)
    angles = np.zeros(ntrials)
    if(keep_orig):
        print "Index is "+str(index)
        print "Mol is "+str(mol[index])+" Length is "+str(len(mol))
        angles[-1]=mcr.calc_dih_angle(np.array([get_atom(atoms,mol[index-3]),
                                                        get_atom(atoms,mol[index-2]),
                                                        get_atom(atoms,mol[index-1]),
                                                        get_atom(atoms,mol[index])]))
        ntrials -= 1
    for i in range(ntrials):
        angles[i] = dih_cdf[np.searchsorted(dih_cdf[:,0],rnd.uniform(0,1)),1]
        rot_dih_index(atoms,mol,angles[i],index)
        lmp.command("run 0 post no")
        pair_pe[i] = get_pair_pe(lmp)
        #mol_pe[i] = get_mol_pe(lmp)
    return (pair_pe,angles)

def get_rosenbluth(energies,prev_energy,beta):
    weights = np.exp(-beta*(energies-prev_energy))
    sum_weights = np.sum(weights)
    norm_weights = weights/sum_weights
    return norm_weights

def regrow_chain(lmp,atoms,mol,index,dih_cdf,orig_charges,beta,keep_orig=False):
    turn_off_atoms(lmp,mol[index:])
    prev_energy = get_pair_pe(lmp)
    ntrials=5
    rosenbluth = 1
    for i in range(index,len(mol)):
        turn_on_atoms(lmp,mol[[i]],orig_charges)
        trials =  get_rot_trials(lmp,atoms,mol,dih_cdf,i,ntrials,keep_orig)
        probs =  get_rosenbluth(trials[0],prev_energy,beta)
        if(keep_orig):
            angle = trials[1][-1]
            rot_dih_index(atoms,mol,angle,i)
            rosenbluth*=trials[0][-1]
        else:
            trial_choice = np.random.choice(np.arange(ntrials),p=probs)
            rot_dih_index(atoms,mol,trials[1][trial_choice],i)
            rosenbluth*=trials[0][trial_choice]
    return rosenbluth

def cbmc(lmp,atoms,mol,index,dih_cdf,orig_charges,beta):
    old_atoms = np.copy(atoms)
    orig_rosen = regrow_chain(lmp,atoms,mol,index,dih_cdf,orig_charges,beta,keep_orig=True)
    new_rosen = regrow_chain(lmp,atoms,mol,index,dih_cdf,orig_charges,beta)
    prob = new_rosen/orig_rosen
    if(rnd.uniform(0,1)>prob):
        mcr.update_coords(old_atoms,lmp)
        return False
    else:
        return True

def get_end2end(atoms,mol):
    first_atom = get_atom(atoms,mol[0])
    last_atom = get_atom(atoms,mol[-1])
    return np.linalg.norm(first_atom[4:7]-last_atom[4:7])

if __name__=='__main__':
    temp = 298.15
    beta = calc_beta(temp)
    dih_cdf = make_dih_cdf('/scratch/snm8xf/np-mc/inputfiles/dih_potential_1',beta)
    max_angle =  0.34906585
    max_radius = 1.4 

    molecule = rdlmp.readAll('/scratch/snm8xf/np-mc/lt_files/nanoparticle/system.data')
    sulfurID = 4
    atoms = molecule[0]
    orig_charges = np.copy(atoms[:,(0,3)])
    sulfurs = atoms[atoms[:,2]==sulfurID][:,0]
    centerRotation = np.array([0.,0.,0.])
    rotationType = 'align'

    molIDs = atoms[np.where(atoms[:,2]==sulfurID)][:,1]
    mols = [rdlmp.getMoleculeAtoms(molecule[1],startID) for startID in sulfurs]
    (ddts,meohs) = mcr.initializeMols(atoms, molecule[1])
    lmp = start_lammps('/scratch/snm8xf/np-mc/lt_files/nanoparticle/system.in')
    get_coords(lmp,atoms)
    numregrowths = 5000
    end2ends = np.zeros(numregrowths)
    for i in range(numregrowths):
        move = rnd.choice(["swap","rotate","shift","cbmc"])
        accepted = False
        if(move=="swap"):
            (ddt,meoh) = (rnd.choice(ddts),rnd.choice(meohs))
            print "Chosen DDT "+str(ddt)+" and MeOH "+str(meoh)
            ddt_id = atoms[(atoms[:,0]==ddt[0])][:,1]
            meoh_id = atoms[(atoms[:,0]==meoh[0])][:,1]
            print "DDT mol ID is "+str(ddt_id)+" and MeOH ID is "+str(meoh_id)
            mcr.swapMolecules(ddt_id[0],meoh_id[0],atoms,centerRotation,rotationType)
            #accepted = True if exp(-beta*(new_energy-old_energy))>rnd.uniform(0,1) else False
        elif(move=="rotate"):
            molId = rnd.choice(molIDs)
            mcr.randomRotate(atoms,molId,max_angle)
        elif(move=="shift"):
            molId = rnd.choice(molIDs)
            randomShift(atoms,molId,max_radius)
        elif(move=="cbmc"):
            mol = rnd.choice(mols)
            index = rnd.choice(np.arange(len(mol)))
            cbmc(lmp,atoms,mol,index,dih_cdf,orig_charges,beta)
            end2ends[i] = get_end2end(atoms,mol)
        
        mcr.update_coords(atoms,lmp)
        dump_atoms(lmp,"regrow.xyz")
    print "Mean of end to end is "+str(np.mean(end2ends))
    print "Std dev of end to end is "+str(np.std(end2ends))
