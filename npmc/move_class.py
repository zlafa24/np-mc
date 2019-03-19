import random as rnd
import numpy as np
import npmc.forcefield_class as ffc
import scipy.special as scm
from math import pi,acos,sin,cos
import npmc.molecule_class as molc

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
        self.kb = 0.0019872041
        self.num_moves = 0
        self.num_accepted = 0

    def get_acceptance_rate(self):
        acceptance_rate = float(self.num_accepted)/float(self.num_moves) if self.num_moves>0 else 0.0
        return acceptance_rate

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
    def __init__(self,simulation,anchortype,numtrials=5,parallel=False):
        self.dihedral_ffs = ffc.initialize_dihedral_ffs(simulation.init_file+'.settings')
        self.numtrials = numtrials
        super(CBMCRegrowth,self).__init__(simulation)
        self.anchortype = anchortype
        self.set_anchor_atoms()
        self.rosen_file = open('Rosenbluth_Data.txt','a')
        self.rosen_file.write('log_Wo\tlog_Wf\tprobability\tNew Energy\taccepted\n')
        self.parallel = parallel
        if parallel:
            self.lmp_clones = [self.simulation.clone_lammps() for i in range(numtrials)]
        self.move_name = "Regrowth"

    def select_random_molecule(self):
        """Selects a random elegible molecule (one with an anchor atom set) from the molecules provided by the Simulation object that the CBMCRegrowth object was passed in at initialization.

        Returns
        -------
        Random Elegible Molecule : Molecule
            A randomly chosen molecule from the list of elegible ones.
        """
        elegible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchortype in [atom.atomType for atom in molecule.atoms])]
        return(rnd.choice(elegible_molecules))

    def set_anchor_atoms(self):
        """For every molecule in the Move's associated simulation the anchor atom is set to the anchortype associated with the CBMCRegrowth instance.  
        Any molecule that does not have an atom of type anchortype is skipped.
        """
        for key, molecule in self.simulation.molecules.items():
            anchorIDs = [atom.atomID for atom in molecule.atoms if atom.atomType==self.anchortype]
            if len(anchorIDs)>0:
                molecule.setAnchorAtom(anchorIDs[0])

    def select_index(self,molecule):
        """Selects a random index of a molecule ranging from 3 to the final index.  The index selection starts at 3 as no dihedral rotation does not move any atoms with index less than 3.

        Parameters
        ----------
        molecule : Molecule
            The Molecule objectone wishes to select a ranomd index from.

        Returns
        -------
        index : int
            A random index from 3 to length of molecule.      
        """
        return(rnd.randrange(3,len(molecule.atoms)))

    def select_dih_angles(self,dih_type):
        """Returns numtrials number of dihedral angles with probability given by the PDF given by the boltzmann distribution determined by the temperature and the dihedral forcefield.

        Parameters
        ----------
        dih_type : int
            An integer which corresponds to the type of Dihedral one wishes to samples.

        Returns
        -------
        trial_dih_angles : Numpy array of floats
            An array of floats which correspond to the selected dihedral angles in radians.
        """
        force_field = [ff for ff in self.dihedral_ffs if ff.dihedral_type==dih_type][0]
        (thetas,ff_pdf) = force_field.get_pdf(self.temp)
        dtheta = thetas[1]-thetas[0]
        trial_dih_angles = np.random.choice(thetas,size=self.numtrials,p=ff_pdf*dtheta)
        return(trial_dih_angles)

    def parallel_evaluate_energies(self,molecule,index,rotations):
        """Evaluates the pair potential energy of numtrials number of rotations about a the dihedral at the given molecule index and returns the result.

        Parameters
        ----------
        molecule : Molecule
            The molecule that is currently being regrown
        index : int
            The index of the monomer in the molecular chain that is being rotated.
        rotations : float
            The degree of rotation of the dihedral in radians needed to reach each sample point for evaluation

        Returns
        -------
        energies : float list
            A list of the pair potential energy for each rotation.
        """
        #coords = [atom.position for atom in self.simulation.atomlist]
        #num_atoms = self.simulation.atoms.shape[0]
        #eval_coords = np.empty((self.numtrials,num_atoms,3))
        coords=[]
        for i,rotation in enumerate(rotations):
            molecule.rotateDihedral(index,rotation)
            coords.append(self.simulation.get_atom_coords())
            #self.simulation.update_clone_coord(self.lmp_clones[i],atom_coords)
            molecule.rotateDihedral(index,-rotation)
        energies = self.simulation.parallel_pair_PE(self.lmp_clones,coords)
        return(np.array([energy for energy in energies]))
        



    def evaluate_energies(self,molecule,index,rotations):
        """Evluates the pair energy of the system for each of the given dihedral rotations at the specified index.  For these enegies to be consistent with CBMC all atoms past the index should be turned off with turn_off_molecule_atoms.
        
        Parameters
        ----------
        molecule : Molecule
            The molecule on which the dihedral rotations will be carried out
        index : int
            The index of the atom that is in the last position of the dihedral to be rotated.
        rotations : list of floats
            A list of floats which represent the desired rotation from the current dihedral angle in Radians.

        Returns
        -------
        energies : Numpy array of floats
            An array of the pair energy for each of the specified rotations.
        """
        energies = np.empty(self.numtrials)
        for i,rotation in enumerate(rotations):
            molecule.rotateDihedral(index,rotation)
            self.simulation.update_coords()
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

    def evaluate_trial_rotations(self,molecule,index,keep_original=False):
        beta = 1./(self.kb*self.temp)
        dihedral = molecule.index2dihedral(index)
        thetas = self.select_dih_angles(dihedral.dihType)
        theta0 = molecule.getDihedralAngle(dihedral)
        rotations = thetas-theta0
        if keep_original:
            rotations[0]=0
        self.turn_off_molecule_atoms(molecule,index-1)
        self.simulation.update_coords()
        initial_energy = self.simulation.get_pair_PE()
        self.turn_off_molecule_atoms(molecule,index)
        energies = self.parallel_evaluate_energies(molecule,index,rotations) if self.parallel else self.evaluate_energies(molecule,index,rotations)
        log_rosen_weight = scm.logsumexp(-beta*(energies-initial_energy))
        log_norm_probs = -beta*(energies-initial_energy)-log_rosen_weight
        try:
            selected_rotation = np.random.choice(rotations,p=np.exp(log_norm_probs))
        except ValueError as e:
            raise ValueError("Probabilities of trial rotations do not sum to 1")
        return(log_rosen_weight,selected_rotation)

    def regrow(self,molecule,index,keep_original=False):
        total_log_rosen_weight = 0
        for index in range(index,len(molecule.atoms)):             
            self.turn_off_molecule_atoms(molecule,index)
            try:
                (log_step_weight,selected_rotation) = self.evaluate_trial_rotations(molecule,index,keep_original)
            except ValueError as e:
                return False
            rotation = 0 if keep_original else selected_rotation
            molecule.rotateDihedral(index,rotation)
            total_log_rosen_weight+=log_step_weight
        return total_log_rosen_weight

    def move(self):
        molecule = self.select_random_molecule()
        index = self.select_index(molecule) 
        log_Wo = self.regrow(molecule,index,keep_original=True)
        log_Wf = self.regrow(molecule,index)
        probability = min(1,np.exp(log_Wf-log_Wo))
        accepted = probability>rnd.random()
        if log_Wo==False or log_Wf==False:
            accepted=False
        self.simulation.update_coords()
        self.num_moves+=1
        if(accepted):
            self.num_accepted+=1
        return(accepted)


class CBMCSwap(CBMCRegrowth):
    def __init__(self,simulation,anchortype,type_lengths,starting_index=3,numtrials=5,parallel=False):
        super(CBMCSwap,self).__init__(simulation,anchortype,numtrials,parallel)
        self.starting_index=starting_index
        self.type1_numatoms, self.type2_numatoms = type_lengths
        self.move_name="CBMCSwap"
    
    def align_molecules(self,molecule1,molecule2):
        molecule1_vector = molecule1.get_com()-molecule1.anchorAtom.position
        molecule2_vector = molecule2.get_com()-molecule2.anchorAtom.position
        molecule1.align_to_vector(molecule2_vector)
        molecule2.align_to_vector(molecule1_vector)

    def swap_anchor_positions(self,molecule1,molecule2):
        anchor_atom1 = [atom for atom in molecule1.atoms if atom.atomType == self.anchortype][0]
        anchor_atom2 = [atom for atom in molecule2.atoms if atom.atomType == self.anchortype][0]
        move1 = anchor_atom2.position - anchor_atom1.position
        move2 = anchor_atom1.position - anchor_atom2.position
        molecule1.move_atoms(move1)
        molecule2.move_atoms(move2)
        self.simulation.update_coords()

    def align_mol_to_positions(self,mol,positions):
        """swap_atom1_positions = np.copy(np.array([mol1.getAtomByMolIndex(i).position for i in range(0,index+1)]))
        swap_atom2_positions = np.copy(np.array([mol2.getAtomByMolIndex(i).position for i in range(0,index+1)]))
        anchor_distance = np.copy(mol1.getAtomByMolIndex(0).position - mol2.getAtomByMolIndex(0).position)

        for i in range(index+1):
            move1 = swap_atom2_positions[i]-mol1.getAtomByMolIndex(i).position
            move2 = swap_atom1_positions[i]-mol2.getAtomByMolIndex(i).position
            mol1.move_atoms_by_index(move1,i)
            mol2.move_atoms_by_index(move2,i)
        """
        for i,position in enumerate(positions):
            move = position-mol.getAtomByMolIndex(i).position
            mol.move_atoms_by_index(move,i)

    def swap_molecule_positions(self,mol1,mol2):
        positions_mol1 = np.copy(np.array([mol1.getAtomByMolIndex(i).position for i in range(self.starting_index+1)]))
        positions_mol2 = np.copy(np.array([mol2.getAtomByMolIndex(i).position for i in range(self.starting_index+1)]))
        self.swap_anchor_positions(mol1,mol2)
        self.align_molecules(mol1,mol2)
        self.align_mol_to_positions(mol1,positions_mol2)
        self.align_mol_to_positions(mol2,positions_mol1)
        self.simulation.update_coords()
         

    def select_random_molecules(self):
        """Selects a random elegible molecule (one with an anchor atom set) from the molecules provided by the Simulation object that the CBMCRegrowth object was passed in at initialization.

        Returns
        -------
        Random Elegible Molecule : Molecule
            A randomly chosen molecule from the list of elegible ones.
        """
        type1_molecules = [molecule for key,molecule in self.molecules.items() if len(molecule.atoms)==self.type1_numatoms]
        type2_molecules = [molecule for key,molecule in self.molecules.items() if len(molecule.atoms)==self.type2_numatoms]
        random_mol_type1 = rnd.choice(type1_molecules)
        random_mol_type2 = rnd.choice(type2_molecules)
        return((random_mol_type1,random_mol_type2))

    def move(self):
        mol1,mol2 = self.select_random_molecules()
        log_Wo_chain1 = self.regrow(mol1,self.starting_index,keep_original=True)
        log_Wo_chain2 = self.regrow(mol2,self.starting_index,keep_original=True)
        self.swap_molecule_positions(mol1,mol2)
        log_Wf_chain1 = self.regrow(mol1,self.starting_index,keep_original=False)
        log_Wf_chain2 = self.regrow(mol2,self.starting_index,keep_original=False)
        probability = min(1,np.exp(log_Wf_chain1+log_Wf_chain2-(log_Wo_chain1+log_Wo_chain2)))
        accepted = probability>rnd.random()
        if not all([log_Wo_chain1,log_Wo_chain2,log_Wf_chain2,log_Wf_chain1]):
            accepted=False
        self.simulation.update_coords()
        self.num_moves+=1
        if(accepted):
            self.num_accepted+=1
        return(accepted)


class TranslationMove(Move):
    def __init__(self,simulation,max_disp,stationary_types):
        super(TranslationMove,self).__init__(simulation)
        self.max_disp = max_disp
        self.stationary_types = set(stationary_types)
        self.move_name = "Translation"

    def translate(self,molecule,displacement):
        molecule.move_atoms(displacement)
        self.simulation.update_coords()

    def select_random_molecule(self):
        """Selects a random elegible molecule (one with no atoms of the type specified by stationary_types) from the molecules provided by the Simulation object that the CBMCRegrowth object was passed in at initialization.

        Returns
        -------
        Random Elegible Molecule : Molecule
            A randomly chosen molecule from the list of elegible ones.
        """
        elegible_molecules = [molecule for key,molecule in self.molecules.items() if not (self.stationary_types.intersection([atom.atomType for atom in molecule.atoms]))]
        return(rnd.choice(elegible_molecules))

    def get_random_move(self):
        theta = 2*pi*rnd.random()
        phi = acos(2*rnd.random()-1)
        r = self.max_disp*rnd.random()
        return(np.array([r*sin(phi)*cos(theta),r*sin(phi)*cos(theta),r*cos(phi)]))

    def move(self):
        beta = 1./(self.kb*self.temp)
        old_energy = self.simulation.get_total_PE()
        molecule = self.select_random_molecule()
        displacement = self.get_random_move()
        self.translate(molecule,displacement)
        new_energy = self.simulation.get_total_PE()
        probability = min(1,np.exp(-beta*(new_energy-old_energy)))
        accepted = probability>rnd.random()
        self.num_moves+=1
        if(accepted):
            self.num_accepted+=1
        return(accepted)


class SwapMove(Move):
    def __init__(self,simulation,anchortype):
        super(SwapMove,self).__init__(simulation)
        self.anchorType = anchortype
        self.move_name = "Swap"

    def get_random_molecules(self):
        elegible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchorType in [atom.atomType for atom in molecule.atoms])]
        return(np.random.choice(elegible_molecules,size=2,replace=False))

    def swap_positions(self,molecule1,molecule2):
        anchor_atom1 = [atom for atom in molecule1.atoms if atom.atomType == self.anchorType][0]
        anchor_atom2 = [atom for atom in molecule2.atoms if atom.atomType == self.anchorType][0]
        move1 = anchor_atom2.position - anchor_atom1.position
        move2 = anchor_atom1.position - anchor_atom2.position
        molecule1.move_atoms(move1)
        molecule2.move_atoms(move2)
        self.simulation.update_coords()

    def align_molecules(self,molecule1,molecule2):
        molecule1_vector = molecule1.get_com()-molecule1.anchorAtom.position
        molecule2_vector = molecule2.get_com()-molecule2.anchorAtom.position
        molecule1.align_to_vector(molecule2_vector)
        molecule2.align_to_vector(molecule1_vector)

    def move(self):
        beta = 1./(self.kb*self.temp)
        old_energy = self.simulation.get_total_PE()
        molecule1,molecule2 = self.get_random_molecules()
        self.align_molecules(molecule1,molecule2)
        self.swap_positions(molecule1,molecule2)
        new_energy = self.simulation.get_total_PE()
        probability = min(1.,np.exp(-beta*(new_energy-old_energy)))
        accepted = rnd.uniform(0,1)<probability
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return(accepted)
        

class RotationMove(Move):
    def __init__(self,simulation,anchortype,max_angle):
        super(RotationMove,self).__init__(simulation)
        self.anchorType = anchortype
        self.max_angle = max_angle
        self.move_name = "Rotation"

    def get_random_molecule(self):
        elegible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchorType in [atom.atomType for atom in molecule.atoms])]
        return(np.random.choice(elegible_molecules))

    def get_molecule_vector(self,molecule):
        return(molecule.get_com()-molecule.anchorAtom.position)

    def get_random_axis(self):
        x_axis,y_axis,z_axis = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
        random_axis = rnd.choice([x_axis,y_axis,z_axis])
        return(random_axis)

    def get_random_angle(self):
        magnitude = rnd.uniform(1e-8,self.max_angle)
        sign = rnd.choice([-1,1])
        return(magnitude*sign)

    def rotate_molecule(self,molecule):
        molecule_vector = self.get_molecule_vector(molecule)
        random_axis = self.get_random_axis()
        random_angle = self.get_random_angle()
        new_vector = molc.rot_quat(molecule_vector,random_angle,random_axis)
        molecule.align_to_vector(new_vector)
        self.simulation.update_coords()

    def move(self):
        beta = 1./(self.kb*self.temp)
        old_energy = self.simulation.get_total_PE()
        molecule = self.get_random_molecule()
        self.rotate_molecule(molecule)
        new_energy = self.simulation.get_total_PE()
        probability = min(1.,np.exp(-beta*(new_energy-old_energy)))
        accepted = rnd.uniform(0,1)<probability
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return(accepted)

















