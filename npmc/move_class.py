import npmc.forcefield_class as ffc
import npmc.molecule_class as molc
import numpy as np
import random as rnd
from math import *
import os
import scipy.special as scm
import scipy.spatial.distance as ssd
import time

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
    
    def set_acceptance_rate_restart(self,index,acceptance_file):
        
        with open(acceptance_file,'rb') as file:
            file.seek(-2,os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
            last_line = file.readline().decode().split()
            self.num_moves += int(last_line[2*index+1].strip())
            self.num_accepted += int(int(last_line[2*index+1].strip()) * float(last_line[2*index+2].strip()))

class CBMCRegrowth(Move):
    """A class that encapsulates a Configurationally Biased Regrowth move as outline by Siepmann et al. that inherits from the Move class.  

    Parameters
    ----------
    dihedral_ffs : list of type DihedralForceField
        A list of DihedralForceField objects associated with the simulation.
    angle_ffs : list of type AngleForceField
        A list of AngleForceField objects associated with the simulation.        
    branch_pdfs : list of type BranchPDF
        A list of BranchPDF objects associated with the simulation; it contains probabilities associated with evenly divided parts of the two-dihedral phase space
        at a branch point.
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.        
    type_lengths : list of type int
        A list with length equal to the number of ligands types and containing the number of atoms in each ligand.    
    numtrials : int
        The number of trial placements used at each step of the regrowth move.  Default is 5 trials.
    read_pdf : Boolean, optional
        A Boolean that determines whether branch point probability density functions (PDFs) are read from a .pkl file or are determined at the start of the simulation
        and then written to a .pkl file.
    """
    def __init__(self,simulation,anchortype,type_lengths,numtrials=5,read_pdf=False):
        super(CBMCRegrowth,self).__init__(simulation)
        self.move_name = "Regrowth"
        self.numtrials = numtrials  
        self.anchortype = anchortype   
        self.type1_numatoms,self.type2_numatoms = type_lengths
        
        #self.set_anchor_atoms()      
        pdf_molecules = self.select_molecule_of_each_type()
        self.angle_ffs = ffc.initialize_angle_ffs(simulation.init_file+'.settings')
        self.dihedral_ffs = ffc.initialize_dihedral_ffs(simulation.init_file+'.settings')
        self.improper_ffs = ffc.initialize_improper_ffs(simulation.init_file+'.settings')    
        self.branch_pdfs = ffc.initialize_branch_pdfs(pdf_molecules,self.angle_ffs,self.dihedral_ffs,self.improper_ffs,self.temp,read_pdf)    

    def select_random_molecule(self):
        """Selects a random eligible molecule (one with an anchor atom set) from the molecules provided by the Simulation object that the CBMCRegrowth object was passed 
        in at initialization.

        Returns
        -------
        random eligible molecule : Molecule
            A randomly chosen molecule from the list of eligible ones.
        """
        eligible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchortype in [atom.atomType for atom in molecule.atoms])]
        return rnd.choice(eligible_molecules)

    def get_site_ID(self,anchorAtom):
        
        sites = np.genfromtxt(self.surfsites_file,skip_header=2)
        #print(np.linalg.norm(anchorAtom.position[None,:]-sites[:,1:],axis=1))
        siteID = np.nonzero(np.linalg.norm(anchorAtom.position[None,:]-sites[:,1:],axis=1) < 0.1)[0]
        #print(sites[siteID],anchorAtom.position)
        
        return siteID

    def set_anchor_atoms(self):
        """For every molecule in the Move's associated simulation the anchor atom is set to the anchortype associated with the CBMCRegrowth instance.  
        Any molecule that does not have an atom of type anchortype is skipped.
        """
            
        for key,molecule in self.simulation.molecules.items():
            anchorAtoms = [atom for atom in molecule.atoms if atom.atomType==self.anchortype]
            anchorIDs = [atom.atomID for atom in anchorAtoms] 
            if len(anchorIDs)>0:
                siteID = self.get_site_ID(anchorAtoms[0])
                molecule.setAnchorAtom(anchorIDs[0],siteID)
    
    def select_index(self,molecule):
        """Selects a random index of a molecule ranging from 3 to the final index.  The index selection starts at 3 as the dihedral rotation does not move any atoms with 
        index less than 3.

        Parameters
        ----------
        molecule : Molecule
            The Molecule object from which one wishes to select a random index.

        Returns
        -------
        index : int
            A random index from 3 to length of molecule.      
        """
        return rnd.randrange(3,len(molecule.atoms))
        
    def select_molecule_of_each_type(self):
        """Selects an example molecule of each ligand type based on the number of atoms in the ligand. 
        This will fail if different ligands contain the same number of atoms.
        
        Returns
        -------
        molecule_of_each_type : list of type Molecule
            A list with exactly 1 instance of the Molecule class for each unique ligand.
        """
        type1_molecule = []; type2_molecule = []
        for molID,molecule in self.molecules.items():
            if len(molecule.atoms)==self.type1_numatoms and len(type1_molecule)<1: type1_molecule.append(molecule)
            if len(molecule.atoms)==self.type2_numatoms and len(type2_molecule)<1: type2_molecule.append(molecule)
            if len(type1_molecule)==1 and len(type2_molecule)==1: break
        return type1_molecule+type2_molecule
    
    def align_mol_to_positions(self,mol,positions):
        """Aligns the atoms in mol to positions.
        
        Parameters
        ----------
        mol : Molecule
            A Molecule object to be aligned.
        positions : Numpy array of floats
            An array of the desired Cartesian coordinates for the atoms in mol.
        """
        for i,position in enumerate(positions):
            move = position-mol.getAtomByMolIndex(i).position
            mol.move_atoms_by_index(move,i)

    def select_dih_angles(self,molecule,dihedrals,keep_original=False):
        """Returns numtrials number of dihedral angles with probability given by the PDF given by the boltzmann distribution determined by the temperature 
        and the dihedral forcefields.

        Parameters
        ----------
        dihedrals : list of type Dihedral
            A list of Dihedral objects including all of the dihedrals involved in the current regrowth step.

        Returns
        -------
        trial_dih_angles : Numpy array of floats
            An array of the selected dihedral angles in radians.
        """
        '''
        try: dihedral = dihedrals[0]
        except: dihedral = dihedrals 
        force_field = [ff for ff in self.dihedral_ffs if ff.dihedral_type==dihedral.dihType][0]
        thetas,ff_pdf = force_field.get_pdf(self.temp)
        dtheta = thetas[1]-thetas[0]
        trial_dih_angles = np.random.choice(thetas,size=self.numtrials,p=ff_pdf*dtheta)
        if keep_original: trial_dih_angles[0] = molecule.getDihedralAngle(dihedral)
        energies = force_field.ff_function(trial_dih_angles)
        return trial_dih_angles,energies
        '''
        theta0s = [molecule.getDihedralAngle(dihedral) for dihedral in dihedrals]
        shifts = np.array([theta0s[0]-theta0 for theta0 in theta0s]) % (2*np.pi)
        force_fields = [[ff for ff in self.dihedral_ffs if ff.dihedral_type==dihedral.dihType][0] for dihedral in dihedrals]
        if len(dihedrals) == 1:
            thetas,ff_pdf = force_fields[0].get_pdf(self.temp)
        else:
            thetas,ff_pdf = ffc.get_combined_pdf(force_fields,self.temp,shifts)
        dtheta = thetas[1]-thetas[0]
        trial_dih_angles = np.random.choice(thetas,size=self.numtrials,p=ff_pdf*dtheta)
        if keep_original: trial_dih_angles[0] = molecule.getDihedralAngle(dihedrals[0])
        energies = np.zeros(len(trial_dih_angles))
        for i,force_field in enumerate(force_fields):
            energies += force_field.ff_function(trial_dih_angles-shifts[i])
        return trial_dih_angles,energies
    
    def select_dih_angles_branched(self,molecule,dihedrals,atoms,keep_original=False):
        """Returns numtrials number of dihedral angles with probability given by the PDF given by the boltzmann distribution determined by the temperature, 
        the dihedral forcefields, and the angle forcefields.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object of the current ligand being regrown.        
        dihedrals : list of type Dihedral
            A list containing the two Dihedral objects involved in the regrowth at a branch point.        
        atoms : list of type Atom
            A list containing the two Atom objects that are regrown at the branch point.
        
        Returns
        -------
        angle_pairs : Numpy array of floats
            An array of the selected dihedral angles in radians. The first dimension corresponds to the trial in numtrials,
            and the second dimension corresponds to the dihedrals within each trial.
        """
        branch_atoms = set([atom for dihedral in dihedrals for atom in dihedral.atoms])
        impropers = [improper for improper in molecule.impropers if improper.atoms.issubset(branch_atoms)]
        angles = [angle for angle in molecule.angles if set([atom.atomID for atom in atoms]).issubset(angle.atoms)]
        for branch_pdf in self.branch_pdfs:
            if [branch_pdf.dihFF1.dihedral_type,branch_pdf.dihFF2.dihedral_type]==[dihedral.dihType for dihedral in dihedrals]:
                pdf = branch_pdf
                break
        pdf_array = pdf.pdf; weights = pdf.weights
        angle_pairs = np.empty([self.numtrials,2])
        energies = np.empty([self.numtrials])
        for i in range(self.numtrials):
            index = np.random.choice(weights.size,p=weights)
            if i == 0 and keep_original: angle_pairs[i,:] = [molecule.getDihedralAngle(dihedrals[0]),molecule.getDihedralAngle(dihedrals[1])]
            else:
                angle_pairs[i,0] = np.random.uniform(pdf_array[index,0],pdf_array[index,1]) 
                angle_pairs[i,1] = np.random.uniform(pdf_array[index,2],pdf_array[index,3])
            energies[i] = pdf.angleFF.ff_function(molc.getAngle(molecule.getAtomByID(angles[0].atom1),molecule.getAtomByID(angles[0].atom2),molecule.getAtomByID(angles[0].atom3)))
            energies[i] += pdf.dihFF1.ff_function(angle_pairs[i,0]) + pdf.dihFF2.ff_function(angle_pairs[i,1]) + pdf.impFF.ff_function(molecule.getDihedralAngle(impropers[0]))
        return angle_pairs,energies,pdf
    
    def turn_off_molecule_atoms(self,molecule,index,atomIDs=None):
        """Turn off the atoms in the specified molecule between the specified index and the end of the molecule.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object in which atoms will be turned off.
        index : int
            The index at which atoms will be turned off.
        """
        if atomIDs is None: atomIDs = []
        if (index+1>=len(molecule.atoms)):
            self.simulation.turn_on_all_atoms()
            return
        indices_to_turn_off = np.arange(index+1,len(molecule.atoms))
        atoms = map(molecule.getAtomsByMolIndex,indices_to_turn_off)
        atomIDs_off = [atom.atomID for atom_lists in atoms for atom in atom_lists[0]]
        atomIDs_off = [atomID for atomID in atomIDs_off if atomID not in atomIDs]
        self.simulation.turn_off_atoms_legacy(atomIDs_off)
    
    def get_branch_point_energy(self,molecule,branch_pdf_full):
        
        index,branch_pdf = branch_pdf_full
        branch_atoms,indexed_atoms = molecule.getAtomsByMolIndex(index)
        prev_atoms,indexed_atoms = molecule.getAtomsByMolIndex(index-1)
        if len(prev_atoms) > 1:
            central_atom,indexed_atoms = molecule.getAtomsByMolIndex(index-2)
            imp_atom,indexed_atoms = molecule.getAtomsByMolIndex(index-3)
        else:  
            central_atom = prev_atoms
            imp_atom,indexed_atoms = molecule.getAtomsByMolIndex(index-2)   
        theta = molc.getAngle(branch_atoms[0],central_atom[0],branch_atoms[1])
        improper = [improper for improper in molecule.impropers if improper.atoms == set([branch_atoms[0].atomID,branch_atoms[1].atomID,central_atom[0].atomID,imp_atom[0].atomID])]
        phi = molecule.getDihedralAngle(improper[0])
        
        energy = branch_pdf.angleFF.ff_function(theta)+branch_pdf.impFF.ff_function(phi)     
        return energy
    
    def evaluate_energies(self,molecule,atoms,rotations):
        """Evluates the pair energy of the system for each of the given dihedral rotations for the specified atoms. 
        For these enegies to be consistent with CBMC all atoms past the index should be turned off with turn_off_molecule_atoms.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object on which the dihedral rotations will be carried out.
        atoms : list of type Atom
            The Atom object that is in the last position of the dihedral to be rotated.
        rotations : list of floats
            A list of floats which represent the desired rotation from the current dihedral angle in Radians.

        Returns
        -------
        energies : Numpy array of floats
            An array of the pair energy for each of the specified rotations.
        """
        energies = np.empty(self.numtrials)
        for i,rotation in enumerate(rotations):
            molecule.rotateDihedrals(atoms,rotation)
            self.simulation.update_coords()
            energies[i]=self.simulation.get_pair_PE()
            molecule.rotateDihedrals(atoms,-rotation)
        return energies

    def evaluate_trial_rotations(self,molecule,index,keep_original=False):
        """At the specified index of a Molecule that is being regrown, numtrials rotations are generated for the full set of dihedral angles relevant to the regrowth step.
        Returns the list of rotations, their cumulative Rosenbluth weights, and a selected rotation chosen probabilistically based on the respective energies of each rotated state.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object that is currently being regrown.
        index : int
            the index of the molecule that is currently being regrown.
        keep_original : Boolean, optional
            A Boolean that determines whether the initial state is included in the list of rotations.

        Returns
        -------
        rotations : Numpy array of floats
            An array of the trial dihedral angles in radians.
        log_rosen_weight : float
            The log of the total Rosenbluth weight of all the trial rotations.
        selected_rotation : float or list of type float
            The selected trial rotation or list of rotations, with length equal to the number of dihedrals in the regrowth step.
        """
        dihedrals,atoms = molecule.index2dihedrals(index)
        branch_pdf = None
        if len(atoms) == 1: 
            thetas,dih_energies = self.select_dih_angles(molecule,dihedrals,keep_original)
            theta0 = molecule.getDihedralAngle(dihedrals[0])
            rotations = thetas-theta0
        else: 
            theta_pairs,dih_energies,branch_pdf = self.select_dih_angles_branched(molecule,dihedrals,atoms,keep_original)
            theta0s = [molecule.getDihedralAngle(dihedral) for dihedral in dihedrals]
            rotations = [thetas-theta0s for thetas in theta_pairs]
        try:
            ghost_atoms = list(map(molecule.getAtomsByMolIndex,np.arange(index+1,len(molecule.atoms))))[0][0]
        except: 
            ghost_atoms = []
        self.simulation.turn_off_atoms([atom.atomID for atom in atoms],[atom.atomID for atom in ghost_atoms])       
        energies = self.evaluate_energies(molecule,atoms,rotations)
        log_rosen_weight = scm.logsumexp(-1./(self.kb*self.temp)*energies)
        log_norm_probs = -1./(self.kb*self.temp)*energies-log_rosen_weight
        try:
            choice = np.random.choice(np.arange(self.numtrials),p=np.exp(log_norm_probs))
        except ValueError as e:
            raise ValueError("Probabilities of trial rotations do not sum to 1")
        if keep_original:
            choice = 0   
        return rotations,log_rosen_weight,rotations[choice],energies[choice],dih_energies[choice],branch_pdf

    def regrow(self,molecule,index,keep_original=False):
        """Each atom, or pair of atoms at a branch point, is regrown individually in a consecutive loop starting from the specified index continuing away from the anchor
        atom to the end of the molecule. The cumulative Rosenbluth weight of each regrowth step, including the weights of each trial rotation, are returned as
        the sum of the log of the weights.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object that is currently being regrown.
        index : int
            the index of the molecule at which the regrowth will start.
        keep_original : Boolean, optional
            A Boolean that determines whether the initial state is included in the list of rotations.

        Returns
        -------
        total_log_rosen_weight : float
            The log of the total Rosenbluth weight of all the regrowth steps.
        """
        total_log_rosen_weight = 0
        total_pair_energy = 0
        total_dih_energy = 0
        branched = False
        branch_pdfs = []
        for i,idx in enumerate(range(index,len(molecule.atoms))):
            dihedrals,atoms = molecule.index2dihedrals(idx) 
            if len(atoms) > 1:
                if branched == False: branched = True; continue
                else: branched == False  
            else: branched = False  
            try:
                rotations,log_step_weight,selected_rotation,pair_energy,dih_energy,branch_pdf = self.evaluate_trial_rotations(molecule,idx,keep_original)
            except ValueError as e:
                return False,False,False,[] 
            total_pair_energy += pair_energy
            total_dih_energy += dih_energy
            if branch_pdf is not None: branch_pdfs.append((idx,branch_pdf))
            rotation = rotations[0] if keep_original else selected_rotation
            molecule.rotateDihedrals(atoms,rotation)
            self.simulation.turn_on_all_atoms()
            self.simulation.update_coords()
            total_log_rosen_weight+=log_step_weight  
        return total_log_rosen_weight,total_pair_energy,total_dih_energy,branch_pdfs

    def move(self):
        """A CBMC regrowth move is performed on a random molecule starting from a random index, and it is accepted according to the Metropolis criteria using the Rosenbluth weights.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the regrowth move was accepted.
        """
        molecule = self.select_random_molecule()
        index = self.select_index(molecule)
        log_Wo,initial_pair_energy,initial_dih_energy,branch_pdfs = self.regrow(molecule,index,keep_original=True)
        initial_bp_energy = np.sum([self.get_branch_point_energy(molecule,branch_pdf) for branch_pdf in branch_pdfs])
        log_Wf,final_pair_energy,final_dih_energy,branch_pdfs = self.regrow(molecule,index)
        self.simulation.update_coords()
        final_bp_energy = np.sum([self.get_branch_point_energy(molecule,branch_pdf) for branch_pdf in branch_pdfs])
        probability = min(1,np.exp(log_Wf-log_Wo))
        accepted = probability>rnd.random()
        if log_Wo==False or log_Wf==False:
            accepted=False
        self.num_moves+=1
        if(accepted):
            self.num_accepted+=1
        return accepted,final_pair_energy+final_dih_energy+final_bp_energy-initial_pair_energy-initial_dih_energy-initial_bp_energy

class CBMCSwap(CBMCRegrowth):
    """A class that encapsulates a Configurationally Biased Regrowth move as outline by Siepmann et al. that inherits from the CBMCRegrowth class.  

    Parameters
    ----------
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.        
    type_lengths : list of type int
        A list with length equal to the number of ligands types and containing the number of atoms in each ligand.    
    numtrials : int
        The number of trial placements used at each step of the regrowth move.  Default is 5 trials.
    read_pdf : Boolean, optional
        A Boolean that determines whether branch point probability density functions (PDFs) are read from a .pkl file or are determined at the start of the simulation
        and then written to a .pkl file.
    starting_index : int, optional
        An integer representing the index at which regrowth begins, indexed from the anchor atom.
    """
    def __init__(self,simulation,anchortype,type_lengths,numtrials=5,read_pdf=False,starting_index=3):
        super(CBMCSwap,self).__init__(simulation,anchortype,type_lengths,numtrials,read_pdf)
        self.starting_index=starting_index
        self.type1_numatoms,self.type2_numatoms = type_lengths
        self.move_name="CBMCSwap"

    
    def select_random_molecules(self):
        """Selects a random eligible molecule, one with an anchor atom set, from the molecules provided by the Simulation object that the CBMCSwap object was passed 
        at initialization.

        Returns
        -------
        random_mol_type1 : Molecule
            A randomly chosen Molecule object with type 1.
        random_mol_type2 : Molecule
            A randomly chosen Molecule object with type 2.
        """
        type1_molecules = [molecule for key,molecule in self.molecules.items() if len(molecule.atoms)==self.type1_numatoms]
        type2_molecules = [molecule for key,molecule in self.molecules.items() if len(molecule.atoms)==self.type2_numatoms]
        random_mol_type1 = rnd.choice(type1_molecules)
        random_mol_type2 = rnd.choice(type2_molecules)
        if random_mol_type1 == random_mol_type2: type2_molecules.remove(random_mol_type1); random_mol_type2 = rnd.choice(type2_molecules)
        return random_mol_type1,random_mol_type2
        
    def rotate_partial_molecule(self,mol,angle):
        """Rotates the atoms in mol from self.starting_index+1 to the end of the molecule, away from the anchor atom, by the given angle.
        
        Parameters
        ----------
        mol : Molecule
            A Molecule object to be aligned.
        angle : float
            The angle of rotation in radians.
        """
        positions = np.copy(np.array([mol.getAtomByMolIndex(i).position for i in np.arange(self.starting_index-2,len(mol.atoms))]))
        rotate_angle = angle - get_bond_angle(positions[0],positions[1],positions[2])
        rotation_axis = np.cross(positions[2]-positions[1],positions[1]-positions[0])
        for i in np.arange(self.starting_index,len(mol.atoms)):
            mol.getAtomByMolIndex(i).position = positions[1]+molc.rot_quat(mol.getAtomByMolIndex(i).position-positions[1],rotate_angle,rotation_axis)

    def swap_molecule_positions(self,mol1,mol2): 
        """Swap the positions of the two given molecules. The atoms from the anchor atom to the atom at self.starting_index-1 in each molecule take on the positions and angles 
        of the other molecule. The atoms from self.starting_index to the end of the molecule are adjusted to maintain the pre-swap bond lengths and angles.
            
        Parameters
        ----------
        mol1,mol2 : Molecule
            The Molecule objects that are being swapped.
        """
        angles = [get_bond_angle(mol.getAtomByMolIndex(self.starting_index-2).position,mol.getAtomByMolIndex(self.starting_index-1).position,mol.getAtomByMolIndex(self.starting_index).position) if len(mol.atoms) > self.starting_index else None for mol in [mol1,mol2]]
        positions_mol1 = np.copy(np.array([mol1.getAtomByMolIndex(i).position for i in range(self.starting_index)]))
        positions_mol2 = np.copy(np.array([mol2.getAtomByMolIndex(i).position for i in range(self.starting_index)]))
        self.align_mol_to_positions(mol1,positions_mol2)
        self.align_mol_to_positions(mol2,positions_mol1)
        for i,mol in enumerate([mol1,mol2]):
            if len(mol.atoms) > self.starting_index:
                self.rotate_partial_molecule(mol,angles[i])
        self.simulation.update_coords()
        
    def move(self,mol1=None,mol2=None):
        """A CBMC swap move is performed on a random molecule starting from a random index, and it is accepted according to the Metropolis criteria using the Rosenbluth weights.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the swap move was accepted.
        """
        if mol1 is None and mol2 is None: mol1,mol2 = self.select_random_molecules() 
        atomIDs = [atom.atomID for atom in mol1.atoms] + [atom.atomID for atom in mol2.atoms]
        
        self.simulation.turn_off_atoms(atomIDs,[])
        initial_pair_PE = self.simulation.get_pair_PE()
        self.simulation.turn_on_all_atoms()
        
        log_Wo_chain1,initial_pair_PE1,initial_dih_PE1,branch_pdfs1 = self.regrow(mol1,self.starting_index,keep_original=True)
        log_Wo_chain2,initial_pair_PE2,initial_dih_PE2,branch_pdfs2 = self.regrow(mol2,self.starting_index,keep_original=True)

        initial_bp_PE1 = np.sum([self.get_branch_point_energy(mol1,branch_pdf) for branch_pdf in branch_pdfs1])
        initial_bp_PE2 = np.sum([self.get_branch_point_energy(mol2,branch_pdf) for branch_pdf in branch_pdfs2])   
        
        self.swap_molecule_positions(mol1,mol2)
        log_Wf_chain1,final_pair_PE1,final_dih_PE1,branch_pdfs1 = self.regrow(mol1,self.starting_index,keep_original=False)
        log_Wf_chain2,final_pair_PE2,final_dih_PE2,branch_pdfs2 = self.regrow(mol2,self.starting_index,keep_original=False)
        self.simulation.update_coords()

        final_bp_PE1 = np.sum([self.get_branch_point_energy(mol1,branch_pdf) for branch_pdf in branch_pdfs1])
        final_bp_PE2 = np.sum([self.get_branch_point_energy(mol2,branch_pdf) for branch_pdf in branch_pdfs2]) 
        
        self.simulation.turn_off_atoms(atomIDs,[])
        final_pair_PE = self.simulation.get_pair_PE()
        self.simulation.turn_on_all_atoms()
        
        probability = min(1,np.exp(log_Wf_chain1+log_Wf_chain2-(log_Wo_chain1+log_Wo_chain2)))
        accepted = probability>rnd.random()
        if not all([log_Wo_chain1,log_Wo_chain2,log_Wf_chain2,log_Wf_chain1]):
            accepted=False
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return accepted,final_pair_PE+final_dih_PE1+final_dih_PE2+ final_bp_PE1+ final_bp_PE2-initial_pair_PE-initial_dih_PE1-initial_dih_PE2-initial_bp_PE1-initial_bp_PE2

class CBMCJump(CBMCRegrowth):
    """A class that encapsulates a Configurationally Biased Regrowth move as outline by Siepmann et al. that inherits from the CBMCRegrowth class.  

    Parameters
    ----------
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.        
    type_lengths : list of type int
        A list with length equal to the number of ligands types and containing the number of atoms in each ligand.    
    numtrials : int
        The number of trial placements used at each step of the regrowth move.  Default is 5 trials.
    read_pdf : Boolean, optional
        A Boolean that determines whether branch point probability density functions (PDFs) are read from a .pkl file or are determined at the start of the simulation
        and then written to a .pkl file.
    starting_index : int, optional
        An integer representing the index at which regrowth begins, indexed from the anchor atom.
    """
    def __init__(self,simulation,anchortype,type_lengths,jump_dists,numtrials=5,read_pdf=False,starting_index=3):
        super(CBMCJump,self).__init__(simulation,anchortype,type_lengths,numtrials,read_pdf)
        self.min_dist = jump_dists[0]; self.max_dist = jump_dists[1]
        self.starting_index=starting_index
        self.type1_numatoms,self.type2_numatoms = type_lengths
        self.move_name="CBMCJump"   
    
    def get_tetrahedron(self,origin,vertices):
        """
        Given a list of the xyz coordinates of the vertices of a tetrahedron, 
        return tetrahedron coordinate system
        """
        mat = (np.array(vertices) - origin).T
        tetra = np.linalg.inv(mat)
        return tetra

    def check_inside_tetra(self,position,tetra,origin):
        """
        Takes a single point or array of points, as well as tetra and origin objects returned by 
        the Tetrahedron function.
        Returns a boolean or boolean array indicating whether the point is inside the tetrahedron.
        """
        newp = np.matmul(tetra, (position-origin).T).T
        return np.all(newp>=0, axis=-1) & np.all(newp <=1, axis=-1) & (np.sum(newp, axis=-1) <=1)
    
    def select_random_jump_positions(self):
        delta_r = self.max_dist
        a = 1 - self.min_dist/delta_r
        faces = rnd.choices(list(self.simulation.faces),k=self.simulation.numtrials_jump)
        positions = np.empty(shape=(self.simulation.numtrials_jump,3))
        for i,face in enumerate([self.simulation.faces[face] for face in faces]):
            cos_angle = np.dot(face.normal,face.vertices[0])/(np.linalg.norm(face.normal)*np.linalg.norm(face.vertices[0]))
            valid_position = False
            while valid_position == False:
                super_vertices = face.vertices + (delta_r/cos_angle)*face.vertices/np.linalg.norm(face.vertices,axis=1)[:,None]
                tetra = self.get_tetrahedron(np.array([0.0,0.0,0.0]),super_vertices)
                A = rnd.random(); B = rnd.random(); C = rnd.random()
                if A+B >= 1: A = 1-A; B = 1-B
                surface_position = super_vertices[0,:] + A*(super_vertices[1,:]-super_vertices[0,:]) + B*(super_vertices[2,:]-super_vertices[0,:])
                surface_position = surface_position - a*delta_r*C*face.normal/np.linalg.norm(face.normal)
                if self.check_inside_tetra(surface_position,tetra,np.array([0.0,0.0,0.0])):
                    positions[i,:] = surface_position
                    valid_position = True
        return positions
    
    def rotate_molecule(self,mol,new_position):

        mol_vector = mol.get_com()-mol.anchorAtom.position
        axis = np.cross(mol.anchorAtom.position,new_position)
        angle = np.arccos(np.dot(mol.anchorAtom.position,new_position)/(np.linalg.norm(mol.anchorAtom.position)*np.linalg.norm(new_position)))
        new_vector = molc.rot_quat(mol_vector,angle,axis)
        mol.align_to_vector(new_vector)
    
    def regrow_anchor_position(self,mol,index=0,keep_original=False):
               
        atoms = [atom.atomID for atom in mol.atoms]
        self.simulation.turn_off_atoms(atoms[:index],atoms[index:])
        original_anchorPos = np.copy(mol.anchorAtom.position)
        trial_positions = self.select_random_jump_positions()          
        energies = np.empty(len(trial_positions))   
        for i in range(len(trial_positions)):
            if not keep_original or i != 0:
                self.rotate_molecule(mol,trial_positions[i,:])
                mol.move_atoms(trial_positions[i,:]-mol.anchorAtom.position)
                self.simulation.update_coords()
                energies[i]=self.simulation.get_pair_PE()
                self.rotate_molecule(mol,original_anchorPos)
                mol.move_atoms(original_anchorPos-mol.anchorAtom.position)
            else:
                energies[i]=self.simulation.get_pair_PE()
        log_rosen_weight = scm.logsumexp(-1./(self.kb*self.temp)*energies)
        log_norm_probs = -1./(self.kb*self.temp)*energies-log_rosen_weight

        try:
            choice = np.random.choice(np.arange(len(trial_positions)),p=np.exp(log_norm_probs))
        except ValueError as e:
            raise ValueError("Probabilities of trial rotations do not sum to 1")
        if not keep_original:
            self.rotate_molecule(mol,trial_positions[choice,:])
            mol.move_atoms(trial_positions[choice,:]-mol.anchorAtom.position)  
        self.simulation.turn_on_all_atoms()
        self.simulation.update_coords()
        return log_rosen_weight

    def move(self):
        """A CBMC swap move is performed on a random molecule starting from a random index, and it is accepted according to the Metropolis criteria using the Rosenbluth weights.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the swap move was accepted.
        """     
        mol = self.select_random_molecule()
        atomIDs = [atom.atomID for atom in mol.atoms] 

        self.simulation.turn_off_atoms(atomIDs,[])
        initial_pair_PE = self.simulation.get_pair_PE()
        self.simulation.turn_on_all_atoms()
        log_Wo_anchor = self.regrow_anchor_position(mol,self.starting_index,keep_original=True)
        log_Wo_chain,init_pair_PE,initial_dih_PE,branch_pdfs = self.regrow(mol,self.starting_index,keep_original=True)     
        initial_bp_PE = np.sum([self.get_branch_point_energy(mol,branch_pdf) for branch_pdf in branch_pdfs])    

        log_Wf_anchor = self.regrow_anchor_position(mol,self.starting_index,keep_original=False)
        log_Wf_chain,fin_pair_PE,final_dih_PE,branch_pdfs = self.regrow(mol,self.starting_index,keep_original=False)      
        self.simulation.update_coords()
        
        final_bp_PE = np.sum([self.get_branch_point_energy(mol,branch_pdf) for branch_pdf in branch_pdfs])
        self.simulation.turn_off_atoms(atomIDs,[])
        final_pair_PE = self.simulation.get_pair_PE()
        self.simulation.turn_on_all_atoms()
        
        probability = min(1,np.exp(log_Wf_chain+log_Wf_anchor-log_Wo_chain-log_Wo_anchor))    
        accepted = probability>rnd.random()
        if not all([log_Wo_chain,log_Wf_chain]):
            accepted=False
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return accepted,final_pair_PE+final_dih_PE+final_bp_PE-initial_pair_PE-initial_dih_PE-initial_bp_PE

class TranslationMove(Move):
    """A class that encapsulates a translation move that inherits from the Move class. A single ligand is translated along the nanoparticle surface up to the given 
    maximum displacement.
    
    Parameters
    ----------
    max_disp : float
        The maximum linear distance in nanometers attempted by a translation move. 
    stationary_types : int or list
        The type or types of atoms that will not be translated; usually the nanoparticle atom types.
    """
    def __init__(self,simulation,max_disp,stationary_types,cluster,cluster_cutoff=10.0):
        super(TranslationMove,self).__init__(simulation)
        self.max_disp = max_disp
        self.cutoff = cluster_cutoff
        self.stationary_types = set(stationary_types)
        self.cluster = cluster
        self.faces = np.array([np.append(face.normal,face.d) for key,face in self.simulation.faces.items()])
        self.move_name = "Translation"

    def translate(self,molecule,displacement):
        """Move the atoms in the specified molecule by the given displacement.
        
        Parameters
        ----------
        molecule : Molecule
            A random Molecule object to be translated.
        displacement : Numpy array of floats
            A 1x3 array of the displacement vector to be applied to all atoms in molecule.
        """
        molecule.move_atoms(displacement)
        self.simulation.update_coords()

    def get_random_molecule(self):
        """Selects a random eligible molecule, one with no atoms of the type specified by stationary_types, from the molecules provided by the Simulation object that the 
        TranslationMove object was passed at initialization.

        Returns
        -------
        random eligible molecule : Molecule
            A randomly chosen Molecule object from the list of eligible ones.
        """
        eligible_molecules = [molecule for key,molecule in self.molecules.items() if not (self.stationary_types.intersection([atom.atomType for atom in molecule.atoms]))]
        return rnd.choice(eligible_molecules)

    def get_random_move(self,anchor_pos,max_perp_shift=0.25):
        """
        Returns
        -------
        move : Numpy array of floats
            A 1x3 array of a random vector displacement in Cartesian coordinates with a magnitude between zero and the specified maximum displacement.
        """

        dists_to_face = np.abs(np.add(np.sum(np.multiply(self.faces[:,:3],anchor_pos),axis=1),self.faces[:,3])) / np.linalg.norm(self.faces[:,:3],axis=1)
        nearest_face = self.faces[np.argmin(dists_to_face),:]
        trans_plane = np.array([nearest_face[0],nearest_face[1],nearest_face[2],np.sum(np.multiply(nearest_face[:3],anchor_pos))])
        unit_normal = trans_plane[:3] / np.linalg.norm(trans_plane[:3])
        perp_shift = unit_normal * rnd.uniform(-max_perp_shift,max_perp_shift)
              
        new_anchor_pos = anchor_pos + np.array([self.max_disp,self.max_disp,self.max_disp])
        while np.linalg.norm(anchor_pos-new_anchor_pos) > self.max_disp:
            dim1 = np.argmax(np.abs(trans_plane[:3]))
            dim2 = (dim1+1)%3; dim3 = (dim1+2)%3 
            new_anchor_pos[dim2] = anchor_pos[dim2] + self.max_disp * rnd.uniform(-1,1)
            new_anchor_pos[dim3] = anchor_pos[dim3] + self.max_disp * rnd.uniform(-1,1)
            new_anchor_pos[dim1] = (trans_plane[3]-trans_plane[dim2]*new_anchor_pos[dim2]-trans_plane[dim3]*new_anchor_pos[dim3]) / trans_plane[dim1]
             
        return new_anchor_pos-anchor_pos+perp_shift
    
    def get_uniq_pair_idxs(self,pair_ids,num_sets):
    
        uniq_arr,uniq_idxs,num_uniq = np.unique(pair_ids,axis=0,return_index=True,return_counts=True)
        uniq_idxs = uniq_idxs[num_uniq==num_sets-1]
        return uniq_idxs
    
    def cluster_move(self):
        
        eligible_mols = np.array([molecule for key,molecule in self.simulation.molecules.items() if not self.stationary_types.intersection([atom.atomType for atom in molecule.atoms])])
        all_positions = [np.array([atom.position for atom in molecule.atoms]) for molecule in eligible_mols]
        random_mol = rnd.choice(eligible_mols)
        displacement = self.get_random_move(random_mol.anchorAtom.position)
        linked_mols = [random_mol]
        attempted_mols = [random_mol]
        frt = 1./(self.kb*self.temp) * rnd.uniform(0.0,0.2)
        
        for linked_mol in linked_mols:
            linked_mol_positions = np.array([atom.position for atom in linked_mol.atoms])
            min_distances = np.array([np.amin(ssd.cdist(linked_mol_positions,mol_positions)) for mol_positions in all_positions])
            nearby_mols = eligible_mols[np.nonzero((min_distances > 0.0) & (min_distances < self.cutoff))]
            linkable_mols = [mol for mol in nearby_mols if mol.molID not in [mol.molID for mol in attempted_mols]] 
            attempted_mols.extend(linkable_mols)
            
            for linkable_mol in linkable_mols:               
                on_atoms = np.concatenate(([atom.atomID for atom in linked_mol.atoms],[atom.atomID for atom in linkable_mol.atoms]),axis=0)
                ghost_atoms = np.array([atom.atomID for atom in self.simulation.atomlist if atom.molID not in [linked_mol.molID,linkable_mol.molID]])
                self.simulation.turn_off_atoms(on_atoms,ghost_atoms)
                
                self.simulation.lmp.command('group molatoms intersect all all')
                self.simulation.lmp.command('group molatoms clear')
                self.simulation.lmp.command(f'group molatoms id {" ".join(map(str,[atom.atomID for atom in linked_mol.atoms]))}')
                self.simulation.lmp.command('neigh_modify exclude group molatoms molatoms')
                
                self.simulation.lmp.command('group molatoms2 intersect all all')
                self.simulation.lmp.command('group molatoms2 clear')
                self.simulation.lmp.command(f'group molatoms2 id {" ".join(map(str,[atom.atomID for atom in linkable_mol.atoms]))}')
                self.simulation.lmp.command('neigh_modify exclude group molatoms2 molatoms2')
                
                E_c = self.simulation.get_pair_PE()
                self.simulation.turn_on_all_atoms()
                 
                probability = max(0,1-np.exp(frt*E_c))
                if E_c > -1.0: probability = 0.0
                linked = probability > rnd.random()
                if linked: linked_mols.append(linkable_mol)

        self.simulation.turn_off_atoms([atom.atomID for mol in linked_mols for atom in mol.atoms],[])       
        old_energy = self.simulation.get_total_PE()
        old_ids,old_energies = self.simulation.get_pairedPE_IDs()      
        for mol in linked_mols:
            self.translate(mol,displacement)
        new_energy = self.simulation.get_total_PE()
        new_ids,new_energies = self.simulation.get_pairedPE_IDs() 
        
        old_uniq_idxs = self.get_uniq_pair_idxs(np.vstack((old_ids,old_ids,new_ids)),3)      
        new_uniq_idxs = self.get_uniq_pair_idxs(np.vstack((new_ids,new_ids,old_ids)),3)      
        shared_energies = np.delete(new_energies,new_uniq_idxs) - np.delete(old_energies,old_uniq_idxs)
        old_uniq_nrgs = old_energies[old_uniq_idxs]; new_uniq_nrgs = new_energies[new_uniq_idxs]
        neg_energy = np.sum(shared_energies[np.nonzero(shared_energies < 0)]) + np.sum(new_uniq_nrgs[np.nonzero(new_uniq_nrgs < 0)]) + np.sum(old_uniq_nrgs[np.nonzero(old_uniq_nrgs < 0)])

        self.simulation.turn_on_all_atoms()
        probability = min(1,np.exp(((frt*(neg_energy))-(1./(self.kb*self.temp)))*(new_energy-old_energy)))
        accepted = probability>rnd.random()
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return accepted,new_energy-old_energy,len(linked_mols)
    
    def move(self):
        
        if self.cluster: accepted,energy,links = self.cluster_move()
        else:
            eligible_mols = np.array([molecule for key,molecule in self.simulation.molecules.items() if not self.stationary_types.intersection([atom.atomType for atom in molecule.atoms])])
            random_mol = rnd.choice(eligible_mols)
            displacement = self.get_random_move(random_mol.anchorAtom.position)
 
            self.simulation.turn_off_atoms([atom.atomID for atom in random_mol.atoms],[])       
            old_energy = self.simulation.get_total_PE()     
            self.translate(random_mol,displacement)
            new_energy = self.simulation.get_total_PE()   
            self.simulation.turn_on_all_atoms()
            
            probability = min(1,np.exp((-1./(self.kb*self.temp))*(new_energy-old_energy)))
            accepted = probability>rnd.random()
            self.num_moves+=1
            if accepted:
                self.num_accepted+=1
            energy = new_energy-old_energy
            links = 0
        return accepted,energy

class RotationMove(Move):
    """A class that encapsulates a translation move that inherits from the Move class. A single ligand is translated along the nanoparticle surface up to the given 
    maximum displacement.
    
    Parameters
    ----------
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.    
    max_angle : float
        The maximum rotation in radians attempted by the rotation move. 
    """
    def __init__(self,simulation,anchortype,max_angle):
        super(RotationMove,self).__init__(simulation)
        self.anchorType = anchortype
        self.max_angle = max_angle
        self.move_name = "Rotation"

    def get_random_molecule(self):
        """Selects a random eligible molecule, one with an anchor atom set, from the molecules provided by the Simulation object that the RotationMove object was passed 
        at initialization.

        Returns
        -------
        random eligible molecule : Molecule
            A randomly chosen molecule from the list of eligible ones.
        """
        eligible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchorType in [atom.atomType for atom in molecule.atoms])]
        return np.random.choice(eligible_molecules)

    def get_molecule_vector(self,molecule):
        """
        Parameters
        ----------
        molecule : Molecule
            The Molecule object being rotated.
            
        Returns
        -------
        com_vector : Numpy array of floats
            A 1x3 array of the vector between the anchor atom and the center of mass of molecule.
        """
        com_vector = molecule.get_com()-molecule.anchorAtom.position
        return com_vector

    def get_random_axis(self):
        """
        Returns
        -------
        random_axis : Numpy array of ints
            A 1x3 array of a unit vector corresponding to the x-, y-, or z-axis.
        """
        x_axis,y_axis,z_axis = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
        random_axis = rnd.choice([x_axis,y_axis,z_axis])
        return random_axis

    def get_random_angle(self):
        """
        Returns
        -------
        Random angle : float
            A random angle in radians between ~zero and the given maximum angle.
        """
        magnitude = rnd.uniform(1e-8,self.max_angle)
        sign = rnd.choice([-1,1])
        return magnitude*sign

    def rotate_molecule(self,molecule):
        """Rotates the specified molecule about a random Cartesian axis by a random number of radians between ~zero and the given maximum angle.
        Parameters
        ----------
        molecule : Molecule
            The Molecule object being rotated.
        """
        molecule_vector = self.get_molecule_vector(molecule)
        random_axis = self.get_random_axis()
        random_angle = self.get_random_angle()
        new_vector = molc.rot_quat(molecule_vector,random_angle,random_axis)
        molecule.align_to_vector(new_vector)
        self.simulation.update_coords()

    def move(self):
        """A rotation move is performed on a random molecule with a random amount of rotation, and it is accepted according to the Metropolis criteria.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the rotation move was accepted.
        """
        molecule = self.get_random_molecule()
        self.simulation.turn_off_atoms([atom.atomID for atom in molecule.atoms],[])
        old_energy = self.simulation.get_total_PE()
        self.rotate_molecule(molecule)
        new_energy = self.simulation.get_total_PE()
        self.simulation.turn_on_all_atoms()
        probability = min(1,np.exp(-1./(self.kb*self.temp)*(new_energy-old_energy)))
        accepted = rnd.uniform(0,1)<probability
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return accepted,new_energy-old_energy

class Move_Legacy(object):
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
    
    def set_acceptance_rate_restart(self,index,acceptance_file):
        
        with open(acceptance_file,'rb') as file:
            file.seek(-2,os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
            last_line = file.readline().decode().split()
            self.num_moves += int(last_line[2*index+1].strip())
            self.num_accepted += int(int(last_line[2*index+1].strip()) * float(last_line[2*index+2].strip()))

class CBMCRegrowth_Legacy(Move_Legacy):
    """A class that encapsulates a Configurationally Biased Regrowth move as outline by Siepmann et al. that inherits from the Move class.  
    Parameters
    ----------
    dihedral_ffs : list of type DihedralForceField
        A list of DihedralForceField objects associated with the simulation.
    angle_ffs : list of type AngleForceField
        A list of AngleForceField objects associated with the simulation.        
    branch_pdfs : list of type BranchPDF
        A list of BranchPDF objects associated with the simulation; it contains probabilities associated with evenly divided parts of the two-dihedral phase space
        at a branch point.
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.        
    type_lengths : list of type int
        A list with length equal to the number of ligands types and containing the number of atoms in each ligand.    
    numtrials : int
        The number of trial placements used at each step of the regrowth move.  Default is 5 trials.
    read_pdf : Boolean, optional
        A Boolean that determines whether branch point probability density functions (PDFs) are read from a .pkl file or are determined at the start of the simulation
        and then written to a .pkl file.
    """
    def __init__(self,simulation,anchortype,type_lengths,numtrials=5,read_pdf=False):
        super(CBMCRegrowth_Legacy,self).__init__(simulation)
        self.move_name = "Regrowth"
        self.numtrials = numtrials  
        self.anchortype = anchortype   
        self.type1_numatoms,self.type2_numatoms = type_lengths
        
        #self.set_anchor_atoms()      
        pdf_molecules = self.select_molecule_of_each_type()
        self.dihedral_ffs = ffc.initialize_dihedral_ffs(simulation.init_file+'.settings')
        self.angle_ffs = ffc.initialize_angle_ffs(simulation.init_file+'.settings')   
        self.improper_ffs = ffc.initialize_improper_ffs(simulation.init_file+'.settings') 
        self.branch_pdfs = ffc.initialize_branch_pdfs(pdf_molecules,self.angle_ffs,self.dihedral_ffs,self.improper_ffs,self.temp,read_pdf)

    def select_random_molecule(self):
        """Selects a random eligible molecule (one with an anchor atom set) from the molecules provided by the Simulation object that the CBMCRegrowth object was passed 
        in at initialization.
        Returns
        -------
        random eligible molecule : Molecule
            A randomly chosen molecule from the list of eligible ones.
        """
        eligible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchortype in [atom.atomType for atom in molecule.atoms])]
        return rnd.choice(eligible_molecules)

    def set_anchor_atoms(self):
        """For every molecule in the Move's associated simulation the anchor atom is set to the anchortype associated with the CBMCRegrowth instance.  
        Any molecule that does not have an atom of type anchortype is skipped.
        """
        for key,molecule in self.simulation.molecules.items():
            anchorIDs = [atom.atomID for atom in molecule.atoms if atom.atomType==self.anchortype]
            if len(anchorIDs)>0:
                molecule.setAnchorAtom(anchorIDs[0],anchorIDs[0])

    def select_index(self,molecule):
        """Selects a random index of a molecule ranging from 3 to the final index.  The index selection starts at 3 as the dihedral rotation does not move any atoms with 
        index less than 3.
        Parameters
        ----------
        molecule : Molecule
            The Molecule object from which one wishes to select a random index.
        Returns
        -------
        index : int
            A random index from 3 to length of molecule.      
        """
        return rnd.randrange(3,len(molecule.atoms))
        
    def select_molecule_of_each_type(self):
        """Selects an example molecule of each ligand type based on the number of atoms in the ligand. 
        This will fail if different ligands contain the same number of atoms.
        
        Returns
        -------
        molecule_of_each_type : list of type Molecule
            A list with exactly 1 instance of the Molecule class for each unique ligand.
        """
        type1_molecule = []; type2_molecule = []
        for molID,molecule in self.molecules.items():
            if len(molecule.atoms)==self.type1_numatoms and len(type1_molecule)<1: type1_molecule.append(molecule)
            if len(molecule.atoms)==self.type2_numatoms and len(type2_molecule)<1: type2_molecule.append(molecule)
            if len(type1_molecule)==1 and len(type2_molecule)==1: break
        return type1_molecule+type2_molecule

    def select_dih_angles(self,dihedrals):
        """Returns numtrials number of dihedral angles with probability given by the PDF given by the boltzmann distribution determined by the temperature 
        and the dihedral forcefields.
        Parameters
        ----------
        dihedrals : list of type Dihedral
            A list of Dihedral objects including all of the dihedrals involved in the current regrowth step.
        Returns
        -------
        trial_dih_angles : Numpy array of floats
            An array of the selected dihedral angles in radians.
        """
        try: dih_type = dihedrals[0].dihType
        except: dih_type = dihedrals.dihType 
        force_field = [ff for ff in self.dihedral_ffs if ff.dihedral_type==dih_type][0]
        thetas,ff_pdf = force_field.get_pdf(self.temp)
        dtheta = thetas[1]-thetas[0]
        trial_dih_angles = np.random.choice(thetas,size=self.numtrials,p=ff_pdf*dtheta)
        return trial_dih_angles
    
    def select_dih_angles_branched(self,molecule,dihedrals,atoms):
        """Returns numtrials number of dihedral angles with probability given by the PDF given by the boltzmann distribution determined by the temperature, 
        the dihedral forcefields, and the angle forcefields.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object of the current ligand being regrown.        
        dihedrals : list of type Dihedral
            A list containing the two Dihedral objects involved in the regrowth at a branch point.        
        atoms : list of type Atom
            A list containing the two Atom objects that are regrown at the branch point.
        
        Returns
        -------
        angle_pairs : Numpy array of floats
            An array of the selected dihedral angles in radians. The first dimension corresponds to the trial in numtrials,
            and the second dimension corresponds to the dihedrals within each trial.
        """
        for branch_pdf in self.branch_pdfs:
            if [branch_pdf.dihFF1.dihedral_type,branch_pdf.dihFF2.dihedral_type]==[dihedral.dihType for dihedral in dihedrals]:
                pdf = branch_pdf
                break
        pdf_array = pdf.pdf; weights = pdf.weights
        angle_pairs = np.empty([self.numtrials,2])
        for i in range(self.numtrials):
            index = np.random.choice(weights.size,p=weights)
            angle_pairs[i,0] = np.random.uniform(pdf_array[index,0],pdf_array[index,1])
            angle_pairs[i,1] = np.random.uniform(pdf_array[index,2],pdf_array[index,3])
        return angle_pairs   
    
    def evaluate_energies(self,molecule,atoms,rotations):
        """Evluates the pair energy of the system for each of the given dihedral rotations for the specified atoms. 
        For these enegies to be consistent with CBMC all atoms past the index should be turned off with turn_off_molecule_atoms.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object on which the dihedral rotations will be carried out.
        atoms : list of type Atom
            The Atom object that is in the last position of the dihedral to be rotated.
        rotations : list of floats
            A list of floats which represent the desired rotation from the current dihedral angle in Radians.
        Returns
        -------
        energies : Numpy array of floats
            An array of the pair energy for each of the specified rotations.
        """
        energies = np.empty(self.numtrials)
        for i,rotation in enumerate(rotations):
            molecule.rotateDihedrals(atoms,rotation)
            self.simulation.update_coords()
            energies[i]=self.simulation.get_pair_PE()
            molecule.rotateDihedrals(atoms,-rotation)
        return energies

    def turn_off_molecule_atoms(self,molecule,index,atomIDs=None):
        """Turn off the atoms in the specified molecule between the specified index and the end of the molecule.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object in which atoms will be turned off.
        index : int
            The index at which atoms will be turned off.
        """
        if atomIDs is None: atomIDs = []
        if (index+1>=len(molecule.atoms)):
            self.simulation.turn_on_all_atoms()
            return
        indices_to_turn_off = np.arange(index+1,len(molecule.atoms))
        atoms = map(molecule.getAtomsByMolIndex,indices_to_turn_off)
        atomIDs_off = [atom.atomID for atom_lists in atoms for atom in atom_lists[0]]
        atomIDs_off = [atomID for atomID in atomIDs_off if atomID not in atomIDs]
        self.simulation.turn_off_atoms_legacy(atomIDs_off)

    def evaluate_trial_rotations(self,molecule,index,keep_original=False):
        """At the specified index of a Molecule that is being regrown, numtrials rotations are generated for the full set of dihedral angles relevant to the regrowth step.
        Returns the list of rotations, their cumulative Rosenbluth weights, and a selected rotation chosen probabilistically based on the respective energies of each rotated state.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object that is currently being regrown.
        index : int
            the index of the molecule that is currently being regrown.
        keep_original : Boolean, optional
            A Boolean that determines whether the initial state is included in the list of rotations.
        Returns
        -------
        rotations : Numpy array of floats
            An array of the trial dihedral angles in radians.
        log_rosen_weight : float
            The log of the total Rosenbluth weight of all the trial rotations.
        selected_rotation : float or list of type float
            The selected trial rotation or list of rotations, with length equal to the number of dihedrals in the regrowth step.
        """
        dihedrals,atoms = molecule.index2dihedrals(index)
        if len(atoms) == 1: 
            thetas = self.select_dih_angles(dihedrals)
            theta0 = molecule.getDihedralAngle(dihedrals[0])
            rotations = thetas-theta0
            if keep_original:
                rotations[0]=0
            self.turn_off_molecule_atoms(molecule,index-1)
        else: 
            theta_pairs = self.select_dih_angles_branched(molecule,dihedrals,atoms)
            theta0s = [molecule.getDihedralAngle(dihedral) for dihedral in dihedrals]
            rotations = [thetas-theta0s for thetas in theta_pairs]
            if keep_original:
                rotations[0]=np.array([0,0])
            self.turn_off_molecule_atoms(molecule,index-1)
        self.simulation.update_coords()
        initial_energy = self.simulation.get_pair_PE()
        self.turn_off_molecule_atoms(molecule,index,atomIDs=[atom.atomID for atom in atoms])
        energies = self.evaluate_energies(molecule,atoms,rotations)
        log_rosen_weight = scm.logsumexp(-1./(self.kb*self.temp)*(energies-initial_energy))
        log_norm_probs = -1./(self.kb*self.temp)*(energies-initial_energy)-log_rosen_weight
        try:
            selected_rotation = rotations[np.random.choice(np.arange(self.numtrials),p=np.exp(log_norm_probs))]
        except ValueError as e:
            raise ValueError("Probabilities of trial rotations do not sum to 1")
        return rotations,log_rosen_weight,selected_rotation

    def regrow(self,molecule,index,keep_original=False):
        """Each atom, or pair of atoms at a branch point, is regrown individually in a consecutive loop starting from the specified index continuing away from the anchor
        atom to the end of the molecule. The cumulative Rosenbluth weight of each regrowth step, including the weights of each trial rotation, are returned as
        the sum of the log of the weights.
        
        Parameters
        ----------
        molecule : Molecule
            The Molecule object that is currently being regrown.
        index : int
            the index of the molecule at which the regrowth will start.
        keep_original : Boolean, optional
            A Boolean that determines whether the initial state is included in the list of rotations.
        Returns
        -------
        total_log_rosen_weight : float
            The log of the total Rosenbluth weight of all the regrowth steps.
        """
        total_log_rosen_weight = 0
        branched = 0
        for idx in range(index,len(molecule.atoms)):
            dihedrals,atoms = molecule.index2dihedrals(idx)
            self.turn_off_molecule_atoms(molecule,idx,atomIDs=[atom.atomID for atom in atoms])
            try:
                rotations,log_step_weight,selected_rotation = self.evaluate_trial_rotations(molecule,idx,keep_original)
            except ValueError as e:
                return False
            rotation = rotations[0] if keep_original else selected_rotation
            molecule.rotateDihedrals(atoms,rotation)
            total_log_rosen_weight+=log_step_weight
        return total_log_rosen_weight

    def move(self):
        """A CBMC regrowth move is performed on a random molecule starting from a random index, and it is accepted according to the Metropolis criteria using the Rosenbluth weights.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the regrowth move was accepted.
        """
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
        return accepted


class CBMCSwap_Legacy(CBMCRegrowth_Legacy):
    """A class that encapsulates a Configurationally Biased Regrowth move as outline by Siepmann et al. that inherits from the CBMCRegrowth class.  
    Parameters
    ----------
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.        
    type_lengths : list of type int
        A list with length equal to the number of ligands types and containing the number of atoms in each ligand.    
    numtrials : int
        The number of trial placements used at each step of the regrowth move.  Default is 5 trials.
    read_pdf : Boolean, optional
        A Boolean that determines whether branch point probability density functions (PDFs) are read from a .pkl file or are determined at the start of the simulation
        and then written to a .pkl file.
    starting_index : int, optional
        An integer representing the index at which regrowth begins, indexed from the anchor atom.
    """
    def __init__(self,simulation,anchortype,type_lengths,numtrials=5,read_pdf=False,starting_index=3):
        super(CBMCSwap_Legacy,self).__init__(simulation,anchortype,type_lengths,numtrials,read_pdf)
        self.starting_index=starting_index
        self.type1_numatoms,self.type2_numatoms = type_lengths
        self.move_name="CBMCSwap"

            
    def select_random_molecules(self):
        """Selects a random eligible molecule, one with an anchor atom set, from the molecules provided by the Simulation object that the CBMCSwap object was passed 
        at initialization.
        Returns
        -------
        random_mol_type1 : Molecule
            A randomly chosen Molecule object with type 1.
        random_mol_type2 : Molecule
            A randomly chosen Molecule object with type 2.
        """
        type1_molecules = [molecule for key,molecule in self.molecules.items() if len(molecule.atoms)==self.type1_numatoms]
        type2_molecules = [molecule for key,molecule in self.molecules.items() if len(molecule.atoms)==self.type2_numatoms]
        random_mol_type1 = rnd.choice(type1_molecules)
        random_mol_type2 = rnd.choice(type2_molecules)
        if random_mol_type1 == random_mol_type2: type2_molecules.remove(random_mol_type1); random_mol_type2 = rnd.choice(type2_molecules)
        return random_mol_type1,random_mol_type2

    def align_mol_to_positions(self,mol,positions):
        """Aligns the atoms in mol to positions.
        
        Parameters
        ----------
        mol : Molecule
            A Molecule object to be aligned.
        positions : Numpy array of floats
            An array of the desired Cartesian coordinates for the atoms in mol.
        """
        for i,position in enumerate(positions):
            move = position-mol.getAtomByMolIndex(i).position
            mol.move_atoms_by_index(move,i)
        
    def rotate_partial_molecule(self,mol,angle):
        """Rotates the atoms in mol from self.starting_index+1 to the end of the molecule, away from the anchor atom, by the given angle.
        
        Parameters
        ----------
        mol : Molecule
            A Molecule object to be aligned.
        angle : float
            The angle of rotation in radians.
        """
        positions = np.copy(np.array([mol.getAtomByMolIndex(i).position for i in np.arange(self.starting_index-2,len(mol.atoms))]))
        rotate_angle = angle - get_bond_angle(positions[0],positions[1],positions[2])
        rotation_axis = np.cross(positions[2]-positions[1],positions[1]-positions[0])
        for i in np.arange(self.starting_index,len(mol.atoms)):
            mol.getAtomByMolIndex(i).position = positions[1]+molc.rot_quat(mol.getAtomByMolIndex(i).position-positions[1],rotate_angle,rotation_axis)

    def swap_molecule_positions(self,mol1,mol2): 
        """Swap the positions of the two given molecules. The atoms from the anchor atom to the atom at self.starting_index-1 in each molecule take on the positions and angles 
        of the other molecule. The atoms from self.starting_index to the end of the molecule are adjusted to maintain the pre-swap bond lengths and angles.
            
        Parameters
        ----------
        mol1,mol2 : Molecule
            The Molecule objects that are being swapped.
        """
        angles = [get_bond_angle(mol.getAtomByMolIndex(self.starting_index-2).position,mol.getAtomByMolIndex(self.starting_index-1).position,mol.getAtomByMolIndex(self.starting_index).position) if len(mol.atoms) > self.starting_index else None for mol in [mol1,mol2]]
        positions_mol1 = np.copy(np.array([mol1.getAtomByMolIndex(i).position for i in range(self.starting_index)]))
        positions_mol2 = np.copy(np.array([mol2.getAtomByMolIndex(i).position for i in range(self.starting_index)]))
        self.align_mol_to_positions(mol1,positions_mol2)
        self.align_mol_to_positions(mol2,positions_mol1)
        for i,mol in enumerate([mol1,mol2]):
            if len(mol.atoms) > self.starting_index:
                self.rotate_partial_molecule(mol,angles[i])
        self.simulation.update_coords()


    def move(self):
        """A CBMC swap move is performed on a random molecule starting from a random index, and it is accepted according to the Metropolis criteria using the Rosenbluth weights.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the swap move was accepted.
        """
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
        if accepted:
            self.num_accepted+=1
        return accepted


class TranslationMove_Legacy(Move_Legacy):
    """A class that encapsulates a translation move that inherits from the Move class. A single ligand is translated along the nanoparticle surface up to the given 
    maximum displacement.
    
    Parameters
    ----------
    max_disp : float
        The maximum linear distance in nanometers attempted by a translation move. 
    stationary_types : int or list
        The type or types of atoms that will not be translated; usually the nanoparticle atom types.
    """
    def __init__(self,simulation,max_disp,stationary_types):
        super(TranslationMove_Legacy,self).__init__(simulation)
        self.max_disp = max_disp
        self.stationary_types = set(stationary_types)
        self.move_name = "Translation"

    def translate(self,molecule,displacement):
        """Move the atoms in the specified molecule by the given displacement.
        
        Parameters
        ----------
        molecule : Molecule
            A random Molecule object to be translated.
        displacement : Numpy array of floats
            A 1x3 array of the displacement vector to be applied to all atoms in molecule.
        """
        molecule.move_atoms(displacement)
        self.simulation.update_coords()

    def get_random_molecule(self):
        """Selects a random eligible molecule, one with no atoms of the type specified by stationary_types, from the molecules provided by the Simulation object that the 
        TranslationMove object was passed at initialization.
        Returns
        -------
        random eligible molecule : Molecule
            A randomly chosen Molecule object from the list of eligible ones.
        """
        eligible_molecules = [molecule for key,molecule in self.molecules.items() if not (self.stationary_types.intersection([atom.atomType for atom in molecule.atoms]))]
        return rnd.choice(eligible_molecules)

    def get_random_move(self):
        """
        Returns
        -------
        move : Numpy array of floats
            A 1x3 array of a random vector displacement in Cartesian coordinates with a magnitude between zero and the specified maximum displacement.
        """
        theta = 2*pi*rnd.random()
        phi = acos(2*rnd.random()-1)
        r = self.max_disp*rnd.random()
        move = np.array([r*sin(phi)*cos(theta),r*sin(phi)*cos(theta),r*cos(phi)])
        return move

    def move(self):
        """A translation move is performed on a random molecule with a random displacement, and it is accepted according to the Metropolis criteria.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the translation move was accepted.
        """
        old_energy = self.simulation.get_total_PE()
        molecule = self.get_random_molecule()
        displacement = self.get_random_move()
        self.translate(molecule,displacement)
        new_energy = self.simulation.get_total_PE()
        probability = min(1,np.exp(-1./(self.kb*self.temp)*(new_energy-old_energy)))
        accepted = probability>rnd.random()
        self.num_moves+=1
        if accepted:
            self.num_accepted+=1
        return accepted

class RotationMove_Legacy(Move_Legacy):
    """A class that encapsulates a translation move that inherits from the Move class. A single ligand is translated along the nanoparticle surface up to the given 
    maximum displacement.
    
    Parameters
    ----------
    anchortype : int
        The type number in the LAMMPS file used for the atom type of the anchor atom in each molecule.    
    max_angle : float
        The maximum rotation in radians attempted by the rotation move. 
    """
    def __init__(self,simulation,anchortype,max_angle):
        super(RotationMove_Legacy,self).__init__(simulation)
        self.anchorType = anchortype
        self.max_angle = max_angle
        self.move_name = "Rotation"

    def get_random_molecule(self):
        """Selects a random eligible molecule, one with an anchor atom set, from the molecules provided by the Simulation object that the RotationMove object was passed 
        at initialization.
        Returns
        -------
        random eligible molecule : Molecule
            A randomly chosen molecule from the list of eligible ones.
        """
        eligible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchorType in [atom.atomType for atom in molecule.atoms])]
        return np.random.choice(eligible_molecules)

    def get_molecule_vector(self,molecule):
        """
        Parameters
        ----------
        molecule : Molecule
            The Molecule object being rotated.
            
        Returns
        -------
        com_vector : Numpy array of floats
            A 1x3 array of the vector between the anchor atom and the center of mass of molecule.
        """
        com_vector = molecule.get_com()-molecule.anchorAtom.position
        return com_vector

    def get_random_axis(self):
        """
        Returns
        -------
        random_axis : Numpy array of ints
            A 1x3 array of a unit vector corresponding to the x-, y-, or z-axis.
        """
        x_axis,y_axis,z_axis = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
        random_axis = rnd.choice([x_axis,y_axis,z_axis])
        return random_axis

    def get_random_angle(self):
        """
        Returns
        -------
        Random angle : float
            A random angle in radians between ~zero and the given maximum angle.
        """
        magnitude = rnd.uniform(1e-8,self.max_angle)
        sign = rnd.choice([-1,1])
        return magnitude*sign

    def rotate_molecule(self,molecule):
        """Rotates the specified molecule about a random Cartesian axis by a random number of radians between ~zero and the given maximum angle.
        Parameters
        ----------
        molecule : Molecule
            The Molecule object being rotated.
        """
        molecule_vector = self.get_molecule_vector(molecule)
        random_axis = self.get_random_axis()
        random_angle = self.get_random_angle()
        new_vector = molc.rot_quat(molecule_vector,random_angle,random_axis)
        molecule.align_to_vector(new_vector)
        self.simulation.update_coords()

    def move(self):
        """A rotation move is performed on a random molecule with a random amount of rotation, and it is accepted according to the Metropolis criteria.
        
        Returns
        -------
        accepted : Boolean
            A Boolean that indicates whether or not the rotation move was accepted.
        """
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
        return accepted


def get_bond_angle(position1,position2,position3):
    """Returns the smallest angle defined by three Cartesian positions.
    
    Parameters
    ----------
    position1,position2,position3 : Numpy arrays of floats
        1x3 arrays of Cartesian coordinates for three locations, where position2 is the central position.
        
    Returns
    -------
    angle : float
        The angle between the vector from position2 to position1 and the vector from position2 to position3.
    """
    vector1 = position2-position1   
    vector2 = position2-position3
    angle = np.arccos(np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))
    return angle















