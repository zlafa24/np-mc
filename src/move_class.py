import random as rnd
import numpy as np
import forcefield_class as ffc

class Move(object):
    """A class used to encapsulate all MC moves that can be used by a simulation.  

    Parameters
    ----------
    lmp : lammps object
        The lammps instance associated with the simulation.  Passed in so that the move can evaluate energies of moves and thereby calculate probabilities of acceptance.

    molecules : list of type Molecule
        A list of molecules that are available to apply a MC move to.

    temp : float
        The temperature of the simulation in Kelvin.
    """
    def __init__(self,simulation):
        self.simulation =simulation
        self.lmp = simulation.lmp
        self.molecules = simulation.molecules
        self.temp = simulation.temp

class CBMCRegrowth(Move):
    """A class that encapsulates a Configurationally Biased Regrowth move as outline by Siepmann et al. that inherits from the Move class.  
    Here the dihedral force fields of the simulation are also passed in to ais in choosing trial positions.

    Parameters
    ----------
    dihedral_ffs : list of type DihedralForceField
        A list of DihedralForceField objects associated with the simulation

    lmp : lammps object
        The lammps instance associated with the simulation.  Passed in so that the move can evaluate energies of moves and thereby calculate probabilities of acceptance.

    molecules : list of type Molecule
        A list of molecules that are available to apply a MC move to.

    numtrials : int
        The number of trial placements used at each step of the regrowth move.  Default is 5 trials.

    anchortype : int
        The type number in the LAMMPS file used for the atom type used for the anchor atom in each molecule.
    """
    def __init__(self,simulation,anchortype,numtrials=5):
        self.dihedral_ffs = ffc.initialize_dihedral_ffs(simulation.init_file+'.settings')
        self.numtrials = numtrials
        super(CBMCRegrowth,self).__init__(simulation)
        self.anchortype = anchortype
        self.set_anchor_atoms()

    def select_random_molecule(self):
        """Selects a random molecule from the molecules provided by the Simulation object that the CBMCRegrowth object was passed in at initialization.
        """
        return(rnd.choice(self.molecules.items())[1])

    def set_anchor_atoms(self):
        for key, molecule in self.simulation.molecules.iteritems():
            anchorID = [atom.atomID for atom in molecule.atoms if atom.atomType==self.anchortype][0]
            molecule.setAnchorAtom(anchorID)

    def select_index(self,molecule):
        return(rnd.randrange(3,len(molecule.atoms)))

    def select_dih_angles(self,dih_type):
        force_field = [ff for ff in self.dihedral_ffs if ff.dihedral_type==dih_type][0]
        (thetas,ff_pdf) = force_field.get_pdf(self.temp)
        dtheta = thetas[1]-thetas[0]
        trial_dih_angles = np.random.choice(thetas,size=self.numtrials,p=ff_pdf*dtheta)
        return(trial_dih_angles)

    def evaluate_energies(self,molecule,index,rotations):
        energies = np.empty(self.numtrials)
        for i,rotation in enumerate(rotations):
            molecule.rotateDihedral(index,rotation)
            self.simulation.update_coords()
            self.simulation.dump_group("all","cbmc_"+str(index)+"_"+str(i)+".xyz")
            energies[i]=self.simulation.get_pair_PE()
            molecule.rotateDihedral(index,-rotation)
        return(energies)

    def turn_off_molecule_atoms(self,molecule,index): 
        if (index+1>=len(molecule.atoms)):
            self.simulation.turn_on_all_atoms()
            return
        indices_to_turn_off = np.arange(index+1,len(molecule.atoms))
        atoms = map(molecule.getAtomByMolIndex,indices_to_turn_off)
        atomIDs = [atom.atomID for atom in atoms]
        self.simulation.turn_off_atoms(atomIDs)

    def regrow(self,molecule,index):
        kb = 0.0019872041
        beta = 1./(kb*self.temp)
        for index in range(index,len(molecule.atoms)):             
            self.turn_off_molecule_atoms(molecule,index)
            dihedral = molecule.index2dihedral(index)
            thetas = self.select_dih_angles(dihedral.dihType)
            theta0 = molecule.getDihedralAngle(dihedral)
            rotations = thetas-theta0
            energies = self.evaluate_energies(molecule,index,rotations)
            probs = np.exp(-beta*energies)
            norm_probs = probs/sum(probs)
            selected_rotation = np.random.choice(rotations,p=norm_probs)
            molecule.rotateDihedral(index,selected_rotation)

    def move(self):
        molecule = self.select_molecule()
        index = self.select_index()












