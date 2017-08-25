import molecule_class as mlc
import numpy as np
import random as rnd

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
        fragments = map(lambda x: self.get_sample_fragment(),range(self.numsamples))
        fragment_types = np.array(map(lambda x : self.get_fragment_category(x),fragments))
        hist, bins = np.histogram(fragment_types,bins=range(0,self.ligands_per_fragment+2),density=True)
        return((hist,bins))



