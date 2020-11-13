"""This module contains the simulation class and associated functions which are meant to encapsulate a LAMMPS simulation.
"""
from lammps import lammps, PyLammps
import npmc.molecule_class as mol
import npmc.atom_class as atm
import npmc.move_class as mvc
import npmc.forcefield_class as ffc
import os
import numpy as np
import random as rnd
from math import *
from subprocess import check_output
import time    


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
    type_lengths : list of type int
        A list with length equal to the number of ligands types and containing the number of atoms in each ligand.    
    nptype : int
        The atom type, as specified in the datafile, for the nanoparticle metal (Ag, Au, etc.).    
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.
    max_disp : float
        The maximum linear distance in nanometers attempted by the translation move.    
    max_angle : float
        The maximum rotation in radians attempted by the rotation move.    
    numtrials : int
        The number of trial rotations for each regrowth step in the configurationally biased moves    
    restart : Boolean, optional
        A Boolean that determines whether this is a new simulation or a restart of a previous simulation.    
    read_pdf : Boolean, optional
        A Boolean that determines whether branch point probability density functions (PDFs) are read from a .pkl file or are determined at the start of the simulation
        and then written to a .pkl file.
    """
    def __init__(self,init_file,datafile,dumpfile,temp,type_lengths=(5,13),nptype=1,anchortype=2,max_disp=1.0,max_angle=0.1745,numtrials=5,moves=[0,1,2,3],seed=None,restart=False,read_pdf=False,legacy=False):
        rnd.seed(seed)
        np.random.seed(seed)
        dname = os.path.dirname(os.path.abspath(init_file))
        print(f'Configuration file is {init_file}')
        print(f'Directory name is {dname}')
        os.chdir(dname)
        self.temp = temp
        self.numtrials = numtrials
        self.molecules = mol.constructMolecules(datafile)
        self.atomlist = self.get_atoms()
        self.move_idxs = moves
        
        self.lmp = lammps("",["-echo","none","-screen","lammps.out"])
        self.lmp.file(os.path.abspath(init_file))
        self.lmp.command("thermo 1")
        self.lmp.command("thermo_style custom step etotal ke temp pe ebond eangle edihed eimp evdwl ecoul elong press")
        self.dumpfile = os.path.abspath(dumpfile)
        self.datafile = os.path.abspath(datafile)
        self.init_file = os.path.abspath(init_file)
        self.exclude=False
        
        self.initializeGroups(self.lmp)
        self.initializeComputes(self.lmp)
        self.initializeFixes(self.lmp)
        self.initializeMoves(type_lengths,nptype,anchortype,max_disp,max_angle,numtrials,read_pdf,restart,legacy)
        self.initialize_data_files(restart)
        self.step = 0 if not restart else self.get_last_step_number()
        self.update_neighbor_list()
        self.deltaE = 0.0

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
        lmp.command("compute imp_pe all pe improper")
        lmp.command("compute ang_pe all pe angle")
        lmp.command("compute bond_pe all pe bond")
        lmp.command("compute coul_pe all pair lj/cut/coul/debye ecoul")
        lmp.command("compute lj_pe all pair lj/cut/coul/debye evdwl")
        lmp.command("compute pair_total all pair lj/cut/coul/debye")

    def initializeFixes(self,lmp):
        """Initializes the LAMMPS fixes one wishes to use in the simulation.
        """
        lmp.command("fix fxfrc silver setforce 0. 0. 0.")
    
    def initializeMoves(self,type_lengths,nptype,anchortype,max_disp,max_angle,numtrials,read_pdf,restart=False,legacy=False):
        """Initializes the Monte Carlo moves used in the simulation.
        """
        translate_move_legacy = mvc.TranslationMove_Legacy(self,max_disp,[nptype])
        rotation_move_legacy = mvc.RotationMove_Legacy(self,anchortype,max_angle)
        cbmc_move_legacy = mvc.CBMCRegrowth_Legacy(self,anchortype,type_lengths,numtrials,read_pdf)
        swap_move_legacy = mvc.CBMCSwap_Legacy(self,anchortype,type_lengths,numtrials,read_pdf)
        translate_move = mvc.TranslationMove(self,max_disp,[nptype])
        rotation_move = mvc.RotationMove(self,anchortype,max_angle)
        cbmc_move = mvc.CBMCRegrowth(self,anchortype,type_lengths,numtrials,read_pdf)
        swap_move = mvc.CBMCSwap(self,anchortype,type_lengths,numtrials,read_pdf)
        self.moves = [cbmc_move,translate_move,swap_move,rotation_move]
        if legacy: self.moves = [cbmc_move_legacy,translate_move_legacy,swap_move_legacy,rotation_move_legacy] 
        self.moves = [self.moves[i] for i in self.move_idxs]
        if restart:
            for i,move in enumerate(self.moves):
                move.set_acceptance_rate_restart(i,'Acceptance_Rate.txt')
    
    def initialize_data_files(self,restart=False):
        """Initializes the potential energy and acceptance rate data files for the simulation.
        """
        if not restart:
            with open('Potential_Energy.txt','w') as potential_file, open('Acceptance_Rate.txt','w') as acceptance_file:
                potential_file.write('step\tEnergy\tmove\tAccepted?\n')
                acceptance_file.write('step\t'+'\t'.join(['#'+move.move_name+'\tRate' for move in self.moves])+'\n')
        self.potential_file = open('Potential_Energy.txt','a') 
        self.acceptance_file = open('Acceptance_Rate.txt','a')

    def get_last_step_number(self):
        """Gets the last simulation step number from the potential energy data file.
        """
        last_line = check_output(["tail","-1",self.potential_file.name])
        return int(last_line.decode().split('\t')[0])
        
    def minimize(self,force_tol=1e-3,e_tol=1e-5,max_iter=500,style='quickmin'):
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
        self.lmp.command('timestep 0.5')
        self.lmp.command(f'min_style {style}')
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
        """Compute the Coulombic potential energy from LAMMPS.
        """
        self.lmp.command("run 0 post no")
        return self.lmp.extract_compute("coul_pe",0,0)

    def getVdwlPE(self):
        """Compute the Van Der Waals potential energy from LAMMPS.
        """
        self.lmp.command("run 0 post no")
        return self.lmp.extract_compute("lj_pe",0,0)
    
    def getDihPE(self):
        """Compute the Van Der Waals potential energy from LAMMPS.
        """
        self.lmp.command("run 0 post no")
        return self.lmp.extract_compute("mol_pe",0,0)

    def get_pair_PE(self):
        """Compute the total pair potential energy from LAMMPS.
        """
        self.lmp.command("run 0 post no")
        energies = self.lmp.extract_compute("pair_total",0,0)
        return energies
    
    def get_total_PE(self):
        """Compute the total potential energy from LAMMPS.
        """
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
        """Update the LAMMPS neighbor list.
        """
        self.lmp.command("run 0 post no")

    def turn_on_all_atoms(self):
        """Turns on short range interactions for all atoms by including all the atoms in the LAMMPS neighbor list.
        """
        self.lmp.command("neigh_modify exclude none")
        if self.exclude:
            self.exclude_type(self.excluded_types[0],self.excluded_types[1])
        self.update_neighbor_list()

    def turn_off_atoms(self,atomIDs,ghost_atoms):
        """Turns off short range interactions with specified atoms by excluding those atoms from the LAMMPS neighbor list.

        Parameters
        ----------
        atomIDs : list of type int
            A list of atom IDs of the atoms that will be turned off in the simulation
        """
        stratoms = ' '.join(map(str,map(int,atomIDs)))
        self.lmp.command("neigh_modify exclude none")       
        if len(ghost_atoms) > 0:
            stratoms_ghost = ' '.join(map(str,map(int,ghost_atoms)))
            self.lmp.command("group ghostatoms intersect all all")
            self.lmp.command("group ghostatoms clear")
            self.lmp.command("group ghostatoms id "+stratoms_ghost)
            self.lmp.command("neigh_modify exclude group ghostatoms all")  
        self.lmp.command("group onatoms intersect all all")
        self.lmp.command("group onatoms clear")
        self.lmp.command("group onatoms id "+stratoms)
        self.lmp.command("group offatoms intersect all all")
        self.lmp.command("group offatoms clear")
        self.lmp.command("group offatoms subtract all onatoms")
        self.lmp.command("neigh_modify exclude group offatoms offatoms")
        self.update_neighbor_list()
        
    def turn_off_atoms_legacy(self,atomIDs):
        """Turns off short range interactions with specified atoms by excluding those atoms from the LAMMPS neighbor list.

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
        """Turns off short range interactions with atoms of two specified types by excluding those atoms from the LAMMPS neighbor list.

        Parameters
        ----------
        type1, type2: int
            The atom types with which short range interactions will be turned off.
        """
        self.excluded_types = (type1,type2)
        self.exclude=True
        self.lmp.command("neigh_modify exclude type %d %d" % (type1,type2))

    def getRandomMolecule(self):
        """Returns a randomly selected molecule from the LAMMPS datafile associated with the given instance.
        """
        return rnd.choice(self.molecules)

    def get_atoms(self):
        """Returns a list of atom objects for all atoms in the simulation.
        """
        atomlist = []
        for key in self.molecules:
            atomlist.extend(self.molecules[key].atoms)
        return atomlist

    def get_coords(self):
        """Updates the atom positions for each instance of the atom class with the atom positions from LAMMPS.
        """
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
        atom_coords : Numpy array of floats
            An Nx3 array with each row representing an atom and the columns containing the x, y and z coordinates, respectively.
        """
        atom_coords = np.array([atom.position for atom in self.atomlist])
        return atom_coords

    def update_coords(self):
        """Updates the atom positions in LAMMPS with the atom positions from each instance of the atom class.
        """
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = self.lmp.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(len(self.atomlist))):
            coords[i*3]=self.atomlist[idx].position[0]
            coords[i*3+1]=self.atomlist[idx].position[1]
            coords[i*3+2]=self.atomlist[idx].position[2]
        self.lmp.scatter_atoms("x",1,3,coords)

    def revert_coords(self,old_positions):
        """Reverts the atom positions in LAMMPS to the given previous atom positions from all atom objects.
        
         Parameters
        ----------
        old_positions: Numpy array of floats
            An Nx3 array with each row representing an atom and the columns containing the x, y and z coordinates, respectively.         
        """
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = self.lmp.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(len(old_positions))):
            coords[i*3]=old_positions[idx][0]
            coords[i*3+1]=old_positions[idx][1]
            coords[i*3+2]=old_positions[idx][2]
        self.lmp.scatter_atoms("x",1,3,coords)
        self.get_coords()

    def check_total_energy(self):
        running_deltaE = self.initial_PE+self.deltaE
        actual_totalPE = self.get_total_PE()
        if not isclose(running_deltaE,actual_totalPE,abs_tol=0.001):
            raise Exception('Total energy has deviated.')
        if isclose(running_deltaE,actual_totalPE,abs_tol=1e-6):
            self.deltaE = self.get_total_PE()-self.initial_PE

    def perform_mc_move(self):
        """Randomly selects one of the Monte Carlo moves included in initializeMoves, performs it, and accepts or rejects the move according to the Metropolis criteria.
        
        Returns
        --------
        accepted : Boolean
            A Boolean for whether the move was accepted or not.
        """
        move = rnd.choice(self.moves)
        old_positions = np.copy([atom.position for atom in self.atomlist])
        accepted,energy = move.move()
        if accepted:
            self.deltaE += energy
        else:  
            self.revert_coords(old_positions)
            self.update_coords()
        #print(self.initial_PE+self.deltaE)
        #print(self.get_total_PE())
        self.step+=1
        self.potential_file.write(f'{self.step}\t{self.initial_PE+self.deltaE}\t{move.move_name}\t{accepted}\n')
        self.acceptance_file.write(str(self.step)+"\t"+"\t".join([str(mc_move.num_moves)+"\t"+str(mc_move.get_acceptance_rate()) for mc_move in self.moves])+"\n")
        self.acceptance_file.flush()
        self.potential_file.flush()
        return accepted
    
    def perform_mc_move_legacy(self):
        """Randomly selects one of the Monte Carlo moves included in initializeMoves, performs it, and accepts or rejects the move according to the Metropolis criteria.
        
        Returns
        --------
        accepted : Boolean
            A Boolean for whether the move was accepted or not.
        """
        move = rnd.choice(self.moves)
        old_positions = np.copy([atom.position for atom in self.atomlist])
        accepted = move.move()
        if not accepted: self.revert_coords(old_positions)
        self.step+=1
        self.update_coords()
        new_energy = self.get_total_PE()
        self.potential_file.write(str(self.step)+'\t'+str(new_energy)+'\t'+str(move.move_name)+'\t'+str(accepted)+'\n') 
        self.acceptance_file.write(str(self.step)+"\t"+"\t".join([str(mc_move.num_moves)+"\t"+str(mc_move.get_acceptance_rate()) for mc_move in self.moves])+"\n")
        self.acceptance_file.flush()
        self.potential_file.flush()
        return accepted

