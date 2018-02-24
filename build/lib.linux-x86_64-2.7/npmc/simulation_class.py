"""This module contains the simulation class and associated functions which are meant to encapsulate a LAMMPS simulation.
"""
from lammps import lammps
import molecule_class as mol
import atom_class as atm
import move_class as mvc
import os
import numpy as np
import random as rnd
import forcefield_class as ffc
import multiprocessing as mpc
import dill
from pathos.multiprocessing import ProcessPool
import concurrent.futures as cncf
from math import *
from subprocess import check_output

class Simulation(object):
    """This class encapsulates a LAMMPS simulation including its associated molecules and computes and fixes.

    Parameters
    ----------
    init_file : str
        The file name of the initialization LAMMPS file. This file contains all the force field parameters computes and fixes that one wishes to specify for the LAMMPS simulation.
    datafile : str
        The LAMMPS data file which contains the coordinates of the atoms and bond, angle, and dihedral information.
    dumpfile : str
        The filename of the file which one wishes to dump the XYZ information of the simulation.
    temp : float
        The temperature which one wishes to run the simulation.
    exclude : binary
        A binary value that determines whether any interactions are excluded in the simulation.
    numtrials : int
        The number of trial rotations for each regrowth step in the configurationally biased moves
    restart : binary
        A binary value that determines whether this is a new simulation or a restart of a previous simulation.
    """
    def __init__(self,init_file,datafile,dumpfile,temp,max_disp=1.0,type_lengths=(5,13),numtrials=5,anchortype=2,restart=False,parallel=False):
        dname = os.path.dirname(os.path.abspath(init_file))
        print "Configuration file is "+str(init_file)
        print 'Directory name is '+dname
        os.chdir(dname)

        self.lmp = lammps("",["-echo","none","-screen","lammps.out"])
        self.lmp.file(os.path.abspath(init_file))
        
        #if parrallel:
        #    pool = mpc.Pool()
        #    self.lmp_clones = pool.map(self.clone_lammps,range(numtrials)) 
        #    pool.close()
        self.molecules = mol.constructMolecules(datafile)
        self.atomlist = self.get_atoms()
        
        self.lmp.command("thermo 1")
        self.lmp.command("thermo_style    custom step etotal ke temp pe ebond eangle edihed eimp evdwl ecoul elong press")
        self.temp = temp
        self.dumpfile = os.path.abspath(dumpfile)
        self.datafile = os.path.abspath(datafile)
        self.init_file = os.path.abspath(init_file)
        self.exclude=False

        self.initializeGroups(self.lmp)
        self.initializeComputes(self.lmp)
        self.initializeFixes(self.lmp)
        self.initializeMoves(anchortype,max_disp,type_lengths,numtrials,parallel)
        self.initialize_data_files(restart) 
        self.step=0 if not restart else self.get_last_step_number()
        self.update_neighbor_list()

    def clone_lammps(self):
        lmp2 = lammps("",["-echo","none","-screen","lammps.out"])
        lmp2.file(os.path.abspath(self.init_file))
        lmp2.command("thermo 1")
        lmp2.command("thermo_style  custom step etotal ke temp pe ebond eangle edihed eimp evdwl ecoul elong press")
        self.initializeGroups(lmp2)
        self.initializeComputes(lmp2)
        self.initializeFixes(lmp2)
        return(lmp2)

    def initializeGroups(self,lmp):
        """Initialize the LAMMPS groups that one wishes to use in the simulation.
        """
        lmp.command("group silver type 1")
        lmp.command("group adsorbate type 2 3 4 5 6")

    def initializeComputes(self,lmp):
        """Initializes the LAMMPS computes that one  wishes to use in the simulation.
        """
        lmp.command("compute pair_pe all pe")
        lmp.command("compute mol_pe all pe dihedral")
        lmp.command("compute coul_pe all pair lj/cut/coul/debye ecoul")
        lmp.command("compute lj_pe all pair lj/cut/coul/debye evdwl")
        lmp.command("compute pair_total all pair lj/cut/coul/debye")

    def initializeFixes(self,lmp):
        """Initializes the fixes one wishes to use in the simulation.
        """
        lmp.command("fix fxfrc silver setforce 0. 0. 0.")
    
    def initializeMoves(self,anchortype,max_disp,type_lengths,numtrials,parallel=False):
        cbmc_move = mvc.CBMCRegrowth(self,anchortype,numtrials)
        translate_move = mvc.TranslationMove(self,max_disp,[1])
        swap_move = mvc.CBMCSwap(self,anchortype,type_lengths,parallel=parallel)
        rotation_move = mvc.RotationMove(self,anchortype,0.1745)
        self.moves = [cbmc_move,translate_move,swap_move,rotation_move]
    
    def initialize_data_files(self,restart=False):
        if not restart:
            with open('Potential_Energy.txt','w') as potential_file, open('Acceptance_Rate.txt','w') as acceptance_file:
                potential_file.write("step\tEnergy\tmove\tAccepted?\n")
                acceptance_file.write("step\t"+"\t".join(["#"+move.move_name+"\tRate" for move in self.moves])+"\n")
        self.potential_file = open('Potential_Energy.txt','a') 
        self.acceptance_file = open("Acceptance_Rate.txt",'a')


    def get_last_step_number(self):
        last_line = check_output(["tail","-1",self.potential_file.name])
        return(int(last_line.split('\t')[0]))
        

    def minimize(self,force_tol=1e-3,e_tol=1e-5,max_iter=200):
        """Minimizes the system using LAMMPS minimize function.

        Parameters
        ----------
        force_tol : float, optional
            The force tolerance used in the minimization routine.
        e_tol : float, optional
            The energy tolerance used in the minimization routine.
        max_iter : int, optional
            The maximum allowed iterations in the minimization procedure.
        """
        #self.lmp.command("dump xyzdump all xyz 10 "+str(self.dumpfile))
        self.lmp.command("minimize "+str(force_tol)+" "+str(e_tol)+" "+str(max_iter)+" "+str(max_iter*10))
        self.get_coords()

    def dump_group(self,group_name,filename):
        """Dumps the atoms of the specified group to an XYZ file specified by filename

        Parameters
        ----------
        group_name : str
            The group ID of the group of atoms that one wishes to dump
        filename : str
            The name of the file that the group of atoms will be dumped to.  
            As the specified format is XYZ it is a good idea to append .xyz to the end of the filename.
        """
        self.lmp.command("write_dump "+group_name+" xyz "+filename)

    def dump_atoms(self):
        """Dump the atom XYZ info to the dumpfile specified in the Simulation's dumpfile variable.
        """
        self.lmp.command("write_dump all xyz "+self.dumpfile+" modify append yes")

    def getCoulPE(self):
        self.lmp.command("run 0 post no")
        return self.lmp.extract_compute("coul_pe",0,0)

    def getVdwlPE(self):
        self.lmp.command("run 0 post no")
        return self.lmp.extract_compute("lj_pe",0,0)

    def get_pair_PE(self): 
        self.lmp.command("run 1 pre no post no")
        return(self.lmp.extract_compute("pair_total",0,0))

    def parallel_pair_PE(self,lmps,coords):
        import pdb;pdb.set_trace()
        n = mpc.cpu_count()
        with ProcessPool(n) as executor:
            energies = executor.map(self.get_clone_pair_PE,lmps,coords)
        return energies

    def get_clone_pair_PE(self,lmp2,coords):
        self.update_clone_coords(lmp2,coords)
        lmp2.command("run 0 post no")
        return(lmp2.extract_compute("pair_total",0,0))

    def get_total_PE(self):
        self.lmp.command("run 0 post no")
        return self.lmp.extract_compute("thermo_pe",0,0)

    def assignAtomTypes(self):
        """Assign element names to the atom types in the simulation.
        """
        atoms = atm.loadAtoms(self.datafile)
        atomtypes =np.unique([atom.atomType for atom in atoms])
        atom_type_dict={}
        for atom_type in atomtypes:
            atom_type_dict[atom_type]=raw_input("Enter element name for atom type "+str(atom_type)+": ")
        self.atom_type_dict=atom_type_dict

    def update_neighbor_list(self):
        self.lmp.command("run 0 post no")

    def turn_on_all_atoms(self):
        self.lmp.command("neigh_modify exclude none")
        if self.exclude:
            self.exclude_type(self.excluded_types[0],self.excluded_types[1])
        self.update_neighbor_list()

    def turn_off_atoms(self,atomIDs):
        """Turns off short range interactions with specified atoms using 'neigh_modify exclude' command in LAMMPS

        Parameters
        ----------
        atomIDs : list of type int
            A list of atom IDs of the atoms that will be turned off in the simulation
        """
        stratoms = ' '.join(map(str,map(int,atomIDs)))
        self.lmp.command("group offatoms intersect all all")
        self.lmp.command("group offatoms clear")
        self.lmp.command("group offatoms id "+stratoms)
        self.lmp.command("neigh_modify exclude group offatoms all")
        if self.exclude:
            self.exclude_type(self.excluded_types[0],self.excluded_types[1])
        self.update_neighbor_list()

    def exclude_type(self,type1,type2):
        self.excluded_types = (type1,type2)
        self.exclude=True
        self.lmp.command("neigh_modify exclude type %d %d" % (type1,type2))

    def getRandomMolecule(self):
        """Returns a randomly selected molecule from the LAMMPS datafile associated with the given instance.

        """
        return(rnd.choice(self.molecules))

    def get_atoms(self):
        atomlist = []
        for key in self.molecules:
            atomlist.extend(self.molecules[key].atoms)
        return atomlist

    def get_coords(self):
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = self.lmp.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(len(self.atomlist))):
            self.atomlist[idx].position[0]=float(coords[i*3])
            self.atomlist[idx].position[1]=float(coords[i*3+1])
            self.atomlist[idx].position[2]=float(coords[i*3+2])

    def get_atom_coords(self):
        """Returns positions of each atom in a Nx3 array

        Returns
        --------
        atom_coords : float array
            A Nx3 array with each row representing an atom and the columns containing the x, ym and z coordinates in that order.
        """
        atom_coords = np.array([atom.position for atom in self.atomlist])
        return atom_coords

    def update_coords(self):
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = self.lmp.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(len(self.atomlist))):
            coords[i*3]=self.atomlist[idx].position[0]
            coords[i*3+1]=self.atomlist[idx].position[1]
            coords[i*3+2]=self.atomlist[idx].position[2]
        self.lmp.scatter_atoms("x",1,3,coords)

    def update_clone_coords(self,lmp2,atom_coords):
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = lmp2.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(atom_coords.shape[0])):
            coords[i*3]=atom_coords[idx][0]
            coords[i*3+1]=atom_coords[idx][1]
            coords[i*3+2]=atom_coords[idx][2]
        lmp2.scatter_atoms("x",1,3,coords)


    def revert_coords(self,old_positions):
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = self.lmp.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(len(old_positions))):
            coords[i*3]=old_positions[idx][0]
            coords[i*3+1]=old_positions[idx][1]
            coords[i*3+2]=old_positions[idx][2]
        self.lmp.scatter_atoms("x",1,3,coords)
        self.get_coords()

    def perform_mc_move(self):
        move = rnd.choice(self.moves)
        old_positions = np.copy([atom.position for atom in self.atomlist])
        accepted = move.move()
        if (not accepted):
            self.revert_coords(old_positions)
        self.step+=1
        self.update_coords()
        new_energy = self.get_total_PE()
        self.potential_file.write(str(self.step)+'\t'+str(new_energy)+'\t'+str(move.move_name)+'\t'+str(accepted)+'\n') 
        self.acceptance_file.write(str(self.step)+"\t"+"\t".join([str(mc_move.num_moves)+"\t"+str(mc_move.get_acceptance_rate()) for mc_move in self.moves])+"\n")
        self.acceptance_file.flush()
        self.potential_file.flush()
        return accepted
            


