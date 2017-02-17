from lammps import lammps
import molecule_class as mol
import atom_class as atm
import os
import numpy as np

class Simulation(object):
    def __init__(self,init_file,datafile,dumpfile,temp):
        self.lmp = lammps()
        dname = os.path.dirname(init_file)
        print "Configuration file is "+str(init_file)
        print 'Directory name is '+dname
        os.chdir(dname)
        self.lmp.file(init_file)
        self.molecules = mol.constructMolecules(datafile)
        self.lmp.command("thermo 1")
	self.lmp.command("thermo_style	custom step etotal ke temp pe ebond eangle edihed eimp evdwl ecoul elong press")
        self.temp = temp
        self.dumpfile = dumpfile
        self.datafile = datafile
        self.init_file = init_file
        
        self.initializeGroups()
        self.initializeComputes()
        self.initializeFixes()
        self.assignAtomTypes()

    def initializeGroups(self):
        self.lmp.command("group silver type 1")
        self.lmp.command("group adsorbate type 2 3 4 5 6")

    def initializeComputes(self):
        self.lmp.command("compute pair_pe all pe")
        self.lmp.command("compute mol_pe all pe dihedral")

    def initializeFixes(self):
	self.lmp.command("fix fxfrc silver setforce 0. 0. 0.")
        self.lmp.command("fix pe_out all ave/time 1 1 1 c_thermo_pe c_pair_pe file pe.out") 
    
    def minimize(self,force_tol=1e-3,e_tol=1e-5,max_iter=200):
        self.lmp.command("dump xyzdump all xyz 10 "+str(self.dumpfile))
        self.lmp.command("minimize "+str(force_tol)+" "+str(e_tol)+" "+str(max_iter)+" "+str(max_iter*10))

    def dump_atoms(self):
        self.lmp.command("write_dump all xyz "+self.dumpfile+" modify append yes")

    def assignAtomTypes(self):
        atoms = atm.loadAtoms(self.datafile)
        atomtypes =np.unique([atom.atomType for atom in atoms])
        atom_type_dict={}
        for atom_type in atomtypes:
            atom_type_dict[atom_type]=raw_input("Enter element name for atom type "+str(atom_type)+": ")
        self.atom_type_dict=atom_type_dict
