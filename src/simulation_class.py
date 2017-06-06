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
from math import *

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
    """
    def __init__(self,init_file,datafile,dumpfile,temp,anchortype=2):
        dname = os.path.dirname(os.path.abspath(init_file))
        print "Configuration file is "+str(init_file)
        print 'Directory name is '+dname
        os.chdir(dname)

        self.lmp = lammps("",["-echo","none","-screen","lammps.out"])
        self.lmp.file(os.path.abspath(init_file))
        self.molecules = mol.constructMolecules(datafile)
        self.atomlist = self.get_atoms()
        
        self.lmp.command("thermo 1")
	self.lmp.command("thermo_style	custom step etotal ke temp pe ebond eangle edihed eimp evdwl ecoul elong press")
        self.temp = temp
        self.dumpfile = os.path.abspath(dumpfile)
        self.datafile = os.path.abspath(datafile)
        self.init_file = os.path.abspath(init_file)
        self.step=0
        self.exclude=False

        self.initializeGroups()
        self.initializeComputes()
        self.initializeFixes()
        self.initializeMoves(anchortype)
        self.initialize_potential_file()
        #self.assignAtomTypes()

    def initializeGroups(self):
        """Initialize the LAMMPS groups that one wishes to use in the simulation.
        """
        self.lmp.command("group silver type 1")
        self.lmp.command("group adsorbate type 2 3 4 5 6")

    def initializeComputes(self):
        """Initializes the LAMMPS computes that one  wishes to use in the simulation.
        """
        self.lmp.command("compute pair_pe all pe")
        self.lmp.command("compute mol_pe all pe dihedral")
        self.lmp.command("compute coul_pe all pair lj/cut/coul/debye ecoul")
        self.lmp.command("compute lj_pe all pair lj/cut/coul/debye evdwl")

    def initializeFixes(self):
        """Initializes the fixes one wishes to use in the simulation.
        """
        self.lmp.command("fix fxfrc silver setforce 0. 0. 0.")
        self.lmp.command("fix pe_out all ave/time 1 1 1 c_thermo_pe c_pair_pe file pe.out") 
    
    def initializeMoves(self,anchortype):
        cbmc_move = mvc.CBMCRegrowth(self,anchortype)
        self.moves = [cbmc_move]

    def initialize_potential_file(self):
        self.potential_file = open('Potential_Energy.txt','a')
        self.potential_file.write("step\tEnergy\tmove\tAccepted?\n")

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
        self.lmp.command("dump xyzdump all xyz 10 "+str(self.dumpfile))
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
        self.lmp.command("run 0 post no")
        return(self.getVdwlPE()+self.getCoulPE())

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

    def turn_on_all_atoms(self):
        self.lmp.command("neigh_modify exclude none")
        if self.exclude:
            self.exclude_type(self.excluded_types[0],self.excluded_types[1])

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

    def update_coords(self):
        indxs = np.argsort([atom.atomID for atom in self.atomlist],axis=0)
        coords = self.lmp.gather_atoms("x",1,3)
        for idx,i in zip(indxs,range(len(self.atomlist))):
            coords[i*3]=self.atomlist[idx].position[0]
            coords[i*3+1]=self.atomlist[idx].position[1]
            coords[i*3+2]=self.atomlist[idx].position[2]
        self.lmp.scatter_atoms("x",1,3,coords)

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
        self.potential_file.write(str(self.step)+'\t'+str(new_energy)+'\t'+'cbmc'+'\t'+str(accepted)+'\n') 
        self.potential_file.flush()
        self.dump_atoms()
        return accepted
            





