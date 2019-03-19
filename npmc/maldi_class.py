import npmc.molecule_class as mlc
import numpy as np
import random as rnd
import scipy.stats as ss

class MALDISpectrum(object):
    """A class that encapsulates a simulated MALDI spectrum along with it's associated helper functions.

    Parameters
    ----------
    data_file : str
        The filename of a LAMMPS data file that contains the desired molecules.
    anchor_type : int
        An integer indicating the atom type of the anchor atom for the molecules.
    numsamples : int
        The number of random samples of the monolayer taken in order to construct the MALDI spectrum.
    nn_distance : float
        The nearest neighbor distance for ligands in the units associated with the given LAMMPS data file (data_file).
    ligands_per_fragment : int
        The number of ligands that compose a MALDI fragment.
    """

    def __init__(self,data_file,anchor_type,numsamples,nn_distance,ligands_per_fragment,type_lengths):
        self.data_file = data_file
        self.molecules = mlc.constructMolecules(self.data_file)
        self.anchor_type = anchor_type
        mlc.set_anchor_atoms(self.molecules,self.anchor_type)
        self.elegible_molecules = [molecule for key,molecule in self.molecules.items() if (self.anchor_type in [atom.atomType for atom in molecule.atoms])] 

        self.dist_dict = self.make_distance_dict()
        self.type1_numatoms, self.type2_numatoms = type_lengths
        self.numsamples = numsamples
        self.nn_distance = nn_distance
        self.ligands_per_fragment = ligands_per_fragment


    def get_random_molecule(self):
        """Gets a random molecule from the possible list of monolayer ligands as determined by the molecules with anchor atoms.
        """
        return(rnd.choice(self.elegible_molecules))

    def get_nns(self,molecule):
        """Returns the N-1 nearest neighbors of the given molecule where N is defined by the variable ligands_per_fragment.

        Parameters
        ----------
        molecule : Molecule
            The molecule one wishes to get the nearest neighbors of.

        Returns
        -------
        nns : list of Molecule
            A list of the N-1 nearest neighbors where N is defined as ligands_per_fragment.
        """
        #distances = self.get_relative_distances(molecule)
        distances = self.dist_dict[molecule.molID]
        elegible_neighbors = distances[np.where(distances[:,1]<self.nn_distance)]
        return(elegible_neighbors)

    def make_distance_dict(self):
        dist_dict = {}
        for molecule in self.elegible_molecules:
            dist_dict[molecule.molID]=self.get_relative_distances(molecule)
        return(dist_dict)

    def get_relative_distances(self,molecule):
        """Returns the relative distance of every molecule from the given molecule sorted by distance.

        Parameters
        ----------
        molecule : Molecule
            The molecule with which one needs the relative distances.

        Returns
        -------
        distances : list of float
            A list of the relative distances for each molecule sorted by distance.
        """
        num_molecules = len(self.elegible_molecules)
        distances = np.zeros((num_molecules,2))
        for i,mol in enumerate(self.elegible_molecules):
            distances[i,0]=mol.molID
            distances[i,1]=np.linalg.norm(molecule.anchorAtom.position-mol.anchorAtom.position)

        distances = distances[distances[:,1].argsort()]
        return(distances)

    def get_sample_fragment(self):
        random_molecule = self.get_random_molecule()
        neighbors = self.get_nns(random_molecule)
        if len(neighbors)<self.ligands_per_fragment:
            return(False)
        else:
            return(neighbors[0:self.ligands_per_fragment])

    def get_molecule_type(self,molecule):
        if len(molecule.atoms)==self.type1_numatoms:
            return 1
        elif len(molecule.atoms)==self.type2_numatoms:
            return 2
        else:
            raise ValueError("Molecule doesn't match either of the molecule types indicated by type1_numatoms or type2_numatoms")

    def get_fragment_category(self,fragment):
        if not np.any(fragment):
            return -1
        else:
            ligand_types = [self.get_molecule_type(self.molecules[ligand[0]]) for ligand in fragment]
            return(len([1 for i in ligand_types if i==1]))
    
    def get_maldi_spectrum(self):
        fragments = [self.get_sample_fragment() for sample in range(self.numsamples)]
        fragment_types = np.array([self.get_fragment_category(fragment) for fragment in fragments])
        hist, bins = np.histogram(fragment_types,bins=range(0,self.ligands_per_fragment+2),density=True)
        self.hist = hist
        self.bins = bins
        return((hist,bins))

    def get_binomial(self,fraction):
        numtries = self.ligands_per_fragment
        numsuccesses = np.arange(self.ligands_per_fragment+1)
        return(ss.binom.pmf(numsuccesses,numtries,fraction))

    def get_SSR(self):
        numtype1 = len([molecule for molecule in self.elegible_molecules if len(molecule.atoms)==self.type1_numatoms])
        numtype2 = len(self.elegible_molecules)-numtype1
        type1_fraction = float(numtype1)/float(numtype2+numtype1)
        binomial = self.get_binomial(type1_fraction)
        return(sum([(self.hist[i]-binomial[i])**2 for i in range(self.ligands_per_fragment+1)]))

    def get_abs_deviation(self):
        numtype1 = len([molecule for molecule in self.elegible_molecules if len(molecule.atoms)==self.type1_numatoms])
        numtype2 = len(self.elegible_molecules)-numtype1
        type1_fraction = float(numtype1)/float(numtype2+numtype1)
        binomial = self.get_binomial(type1_fraction)
        return(sum([abs(self.hist[i]-binomial[i]) for i in range(self.ligands_per_fragment+1)]))

    def swap_molecules(self,mol1,mol2):
        swap_vector = mol1.anchorAtom.position - mol2.anchorAtom.position
        mol1.move_atoms(-swap_vector)
        mol2.move_atoms(swap_vector)

    def get_sensitivity(self,numtrials=200,numligand_swapped=1,dev_function='SSR'):
        type1_mols = [molecule for molecule in self.elegible_molecules if len(molecule.atoms)==self.type1_numatoms]
        type2_mols = [molecule for molecule in self.elegible_molecules if len(molecule.atoms)==self.type2_numatoms]
        original_SSR = self.get_SSR()
        ssrs = np.empty(numtrials)
        mols_type1 = np.empty(numligand_swapped,dtype=object)
        mols_type2 = np.empty(numligand_swapped,dtype=object)
        for i in range(numtrials):
            if((i+1)%10==0):
                print('On step '+str(i+1))
            for j in range(numligand_swapped):
                mols_type1[j] = rnd.choice(type1_mols)
                mols_type2[j] = rnd.choice(type2_mols)
                self.swap_molecules(mols_type1[j],mols_type2[j])
            self.dist_dict = self.make_distance_dict()
            self.get_maldi_spectrum()
            if(dev_function=='SSR'):
                ssrs[i]=self.get_SSR()
            elif(dev_function=='abs_deviation'):
                ssrs[i]=self.get_abs_deviation()
            for j in range(numligand_swapped):
                self.swap_molecules(mols_type1[j],mols_type2[j])
        return original_SSR,ssrs
