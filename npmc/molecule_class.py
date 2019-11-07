"""This module contains the molecule class and all classes and functions
related to molecules.  This includes Bond, Angle, and Dihedral classes as well as helper functions that take in atom and bond lists and generate molecules.
"""

import npmc.read_lmp_rev6 as rdlmp
import npmc.atom_class as atm
from itertools import groupby,permutations
import networkx as ntwkx
import numpy as np
from math import *

class Molecule(object):
    """This class is used to represent a molecule in a simulation and holds all
        the objects related to a molecule including the related Atom, Bond, Angle, and Dihedral Objects.

        Parameters
        ----------
        molID : int
            The unique integer identifier of the molecule as set in the LAMMPS input file
        atoms : list of type Atom
            The Atom objects associated with this molecule
        bonds : list of type Bond
            The Bond objects associated with this molecule
        angles : list of type Angle
            The Angle objects associated with this molecule
        dihedrals : list of type Dihedral
            The Dihedral objects associated with this molecule
    """
    def __init__(self,molID,atoms,bonds,angles,dihedrals):
        self.molID = molID
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.graph = self.atomsAsGraph() if self.bonds is not None else None

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return all([self.atoms==other.atoms,self.bonds==other.bonds,self.angles==other.angles,self.dihedrals==other.dihedrals])
        else:
            return False
            
    def __neq__(self,other):
        return not self.__eq__(other)

    def get_com(self):
        """Calculates the center of mass of the molecule.

        Returns
        -------
        com : array of floats
            The center of mass of the molecule.
        """
        positions = np.array([atom.position for atom in self.atoms])
        com = np.mean(positions,axis=0)
        return com

    def setAnchorAtom(self,atomID):
        """Sets the anchor atom of the molecule: the atom of the molecule that is anchored to the nanoparticle.  Setting this is important when using the CBMC regrowth move.
        
        Parameters
        ----------
        atomID : int
            The unique atom ID of the Atom object that is anchored to the nanoparticle.
        """
        atom = self.getAtomByID(atomID)
        if(atom!=None):
            self.anchorAtom = atom
            return True
        else:
            return False
    
    def atomsAsGraph(self):
        """Returns the graph data structure of the atoms in the molecule where the connectivity is defined by the Bond objects of the molecule.  
        The nodes of the graph are the atom ID's of the atoms.

        Returns
        -------
        Networkx Graph Object
            A Networkx Graph object where the nodes are the atom ID's of the atoms in the molecule and the connections are defined by the Molecule's Bonds.
        """
        return molecule2graph(self.atoms,self.bonds)
        
    def findBranchPoints(self): 
        """Identifies all of the branch points in the molecule; Branch points occur where an atom is bonded to at least three other atoms.
        
        Returns
        -------
        branch_points : Numpy array of ints
            A 2D array where the first axis corresponds to the individual branch points and the second axis corresponds to the dihedral and bond angle types for each branch point.
        """
        branch_points = []
        predecessor_dict = dict(ntwkx.bfs_predecessors(self.graph,source=self.anchorAtom.atomID))
        successor_dict = dict(ntwkx.bfs_successors(self.graph,source=self.anchorAtom.atomID))
        for node in self.graph:
            if len(self.graph[node])>2: 
                branch_atoms = set(list(self.graph[node].keys())+[node]+[predecessor_dict[predecessor_dict[node]]])
                dihedrals = [dihedral for dihedral in self.dihedrals if dihedral.atoms.issubset(branch_atoms)]
                angle_atoms = list(self.graph[node].keys())+[node]; angle_atoms.remove(predecessor_dict[node])
                angle_type = [angle.angleType for angle in self.angles if angle.atoms==set(angle_atoms)][0]
                angle1 = getAngle(self.getAtomByID(predecessor_dict[node]),self.getAtomByID(node),self.getAtomByID(successor_dict[node][0]))
                angle2 = getAngle(self.getAtomByID(predecessor_dict[node]),self.getAtomByID(node),self.getAtomByID(successor_dict[node][1]))
                branch_points.append(([dihedrals[0].dihType,dihedrals[1].dihType,angle_type],[angle1,angle2]))
        return branch_points
        
    def getAtomByID(self,atomID):
        """Returns the Atom object associated with the given atom ID as long as the atom is associated with the molecule.

        Parameters
        ----------
        atomID : int
            The unique atom ID identifier of the Atom object which you hope to retrieve

        Returns
        -------
        atom : Atom or None
            The Atom object associated with the atom ID; None if the Atom is not associated with this molecule.
        """
        for atom in self.atoms:
            if(atom.atomID==atomID):
                return atom
        return None

    def getAtomByMolIndex(self,index):
        """Returns the Atom by its index in the molecule where the index is defined as the number of bonds away from the anchor atom 
        (i.e. the atom at index 1 is the atom directly connected to the anchor atom)

        Parameters
        ----------
        index : int
            The molecular index of the Atom object one wishes to retrieve where the index is defined as the number of bonds away from the anchor atom.

        Returns
        -------
        Atom Object
            The Atom Object located at the specified index. If the index is greater than the number of atoms in the molecule minus one then 
            the function returns None as this is out of the range of the atom list.
        """
        if(index>(len(self.atoms)-1)):
            return None
        successor_dict = ntwkx.bfs_successors(self.graph,source=self.anchorAtom.atomID)
        currentID = self.anchorAtom.atomID
        idx = 0
        while idx < index:
            level = next(successor_dict)
            while idx < index:
                try: 
                    currentID = level[1].pop(0)
                    idx += 1
                except: break
        return self.getAtomByID(currentID)
        
    def getAtomsByMolIndex(self,index):
        """Returns the Atom by it's index in the molecule where the index is defined as the number of bonds away from the anchor atom 
        (i.e. the atom at index 1 is the atom directly connected to the anchor atom)

        Parameters
        ----------
        index : int
            The molecular index of the Atom object one wishes to retrieve where the index is defined as the number of bonds away from the anchor atom.

        Returns
        -------
        Atom Object
            The Atom Object located at the specified index if the index is greater than the number of atoms in the molecule minus one then 
            the function returns None as this is out of the range of the atom list.
        """
        if(index>(len(self.atoms)-1)):
            return None
        successor_dict = ntwkx.bfs_successors(self.graph,source=self.anchorAtom.atomID)
        currentID = self.anchorAtom.atomID
        indexed_atoms = [currentID]
        idx = 0
        while idx < index:
            atoms = []
            IDs = next(successor_dict)[1]
            indexed_atoms.extend(IDs)
            for ID in IDs:
                atoms.append(self.getAtomByID(ID))
            idx += len(IDs)
        return atoms,indexed_atoms
        
    def rotateDihedrals(self,atoms,thetas):
        """Rotates the atoms of the molecule from the given atom or atoms to the last atom (here the last atom is the one opposite the defined "anchor atom") about a dihedral axis.
        The dihedral axis is defined by the two atoms that come before the given atom or atoms as defined by the graph of the molecule. 
        The length of atoms should be equal to the length of thetas; each Atom object in atoms will be rotated by the corresponding angle in thetas.
        
        Parameters
        ----------
        atoms : list of Atom objects
            The atom or atoms with which to start the rotation and consequently the last atom of the dihedral of whose axis you wish to rotate the molecule about.
        thetas : list of floats
            The angle or angles in radians which specifies the angle of rotation about the dihedral axis which you want to perform.
        """
        successor_dict = dict(ntwkx.bfs_successors(self.graph,source=self.anchorAtom.atomID))
        predecessor_dict = dict(ntwkx.bfs_predecessors(self.graph,source=self.anchorAtom.atomID))
        if len(atoms) == 1: thetas = [thetas]
        for i,atom4 in enumerate(atoms):
            atom3_ID = predecessor_dict[atom4.atomID]
            atom3 = self.getAtomByID(atom3_ID)
            atom2 = self.getAtomByID(predecessor_dict[atom3_ID])
            axis = atom3.position-atom2.position
            rotate_ID = atom4.atomID
            branch_ID = 0
            while True:
                atom4 = self.getAtomByID(rotate_ID)
                vector = atom4.position-atom3.position
                atom4.position = rot_quat(vector,thetas[i],axis)+atom3.position
                try: rotate_ID = successor_dict[rotate_ID]
                except: 
                    if branch_ID > 0: 
                        rotate_ID = branch_ID
                        atom4 = self.getAtomByID(rotate_ID)
                        vector = atom4.position-atom3.position
                        atom4.position = rot_quat(vector,thetas[i],axis)+atom3.position
                        branch_ID = 0
                        try: rotate_ID = successor_dict[rotate_ID]
                        except: break
                    else: break
                if len(rotate_ID) == 1:
                    rotate_ID = rotate_ID[0]
                elif len(rotate_ID) == 2: 
                    branch_ID = rotate_ID[1]
                    rotate_ID = rotate_ID[0]

    def findPaths(self,atom,length):
        """Finds all the sets of atoms of the specified length which contain the given atom and are connected by Bonds.
                
        Parameters
        ----------
        atom : Atom
            The Atom object in which to find paths.
        length : int
            The length of paths to find.
        
        Returns
        -------
        paths : list of lists of Atoms
            A list containing lists of Atom objects for each path of the specified length which includes atom.
        """
        if length == 0:
            return [[atom]]
        paths = [[atom]+path for neighbor in self.graph.neighbors(atom) for path in self.findPaths(neighbor,length-1) if atom not in path]
        return paths
    
    def index2dihedrals(self,index):
        """Returns the dihedral associated with the given atom index of the molecule.

        Parameters
        -----------
        index : int
            The index of the molecule that one wants the associated dihedral.  The associated dihedral is the dihedral where the atom is the fourth atom of the dihedral.

        Returns
        -------
        dihedral : Dihedral
            The dihedral associated with the given index.
        """
        if(index<3 or index>(len(self.atoms)-1)):
            raise ValueError("You must pass the index of the fourth atom, which means the index must br greater than 2 and less than the length of the molecule.  The index you passed %d" % index)
        dihedrals_IDs = []
        dihedrals = []
        atoms,indexed_atoms = self.getAtomsByMolIndex(index)
        for atom in atoms:
            dihedrals_IDs.extend(self.findPaths(atom.atomID,3))
        for dihedral_IDs in list(dihedrals_IDs):
            if any(atom not in indexed_atoms for atom in dihedral_IDs): dihedrals_IDs.remove(dihedral_IDs)
            else: dihedrals.extend([dihedral for dihedral in self.dihedrals if set(dihedral_IDs) == dihedral.atoms])
        return dihedrals,atoms

    
    def getDihedralAngle(self,dihedral):
        """Calculates the dihedral angle of a given dihedral.
        
        Parameters
        ----------
        dihedral : Dihedral 
            Dihedral Object from which to calculate the dihedral angle.

        Returns
        -------
        angle : float
            The current dihedral angle of the Dihedral object.
        """
        atom1 = self.getAtomByID(dihedral.atom1)
        atom2 = self.getAtomByID(dihedral.atom2)
        atom3 = self.getAtomByID(dihedral.atom3)
        atom4 = self.getAtomByID(dihedral.atom4)
        b1 = atom2.position-atom1.position
        b2 = atom3.position-atom2.position
        b3 = atom4.position-atom3.position
        b2norm = b2/np.linalg.norm(b2)
        n1 = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
        n2 = np.cross(b2,b3)/np.linalg.norm(np.cross(b2,b3))
        m1 = np.cross(n1,n2)
        angle = atan2(np.dot(m1,b2norm),np.dot(n1,n2))    
        angle=angle%(2*pi)
        return angle
        
    def align_to_vector(self,vector):
        """Rotates all the atoms in the Molecule to align the vector between the anchor atom and the center of mass to the given vector.
        
        Parameters
        ----------
        vector : Numpy array of floats
            A 1x3 array of the Cartesian vector to which the Molecule will be aligned.
        """
        molecule_vector = self.get_com()-self.anchorAtom.position
        if np.linalg.norm(vector)==0. or np.linalg.norm(molecule_vector)==0.:
            raise ValueError("Alignment vector passed in must have a non-zero magnitude.")
        anchor_position = self.anchorAtom.position
        axis_rotation = np.cross(molecule_vector,vector)
        angle = acos(np.dot(molecule_vector/np.linalg.norm(molecule_vector),vector/np.linalg.norm(vector)))
        rotate_atoms = [atom for atom in self.atoms if not (atom.atomID==self.anchorAtom.atomID)]
        for atom in rotate_atoms:
            atom.position = rot_quat((atom.position-anchor_position),angle,axis_rotation)+anchor_position       

    def move_atoms(self,move):
        """Moves all the atoms in the Molecule according to the given move.
        
        Parameters
        ----------
        move : Numpy array of floats
            A 1x3 array of the Cartesian vector for the move.
        """
        for atom in self.atoms:
            atom.position+=move

    def move_atoms_by_index(self,move,index):
        """Moves the atoms between the specified index and the end of the Molecule (opposite the anchor atom) according to the given move.
        
        Parameters
        ----------
        move : Numpy array of floats
            A 1x3 array of the Cartesian vector for the move.
        index : int
            The index of the Atom at which the move should begin.
        """
        for i in range(index,len(self.atoms)):
            self.getAtomByMolIndex(i).position+=move

class Bond(object):
    """The Bond object represents a bond between two atoms. The format is similar to a LAMMPS bond, therefore a 
    Bond object consists of a Bond ID which uniquely defines the bond, a bond type, and the atom ID's of the two atoms involved in the bond.
    
    Parameters
    ----------
    bondID : int
        The unique integer identifying the bonds same as the one defined in the LAMMPS input file
    bondType : int
        An integer which represents the type of bond this is, the same as the bond type number defined in LAMMPS input file.
    atom1 : int
        The atom ID associated with the first atom in the bond
    atom2 : int
        The atom ID associated with the second atom in the bond
    """
    def __init__(self,bondID,bondType,atom1,atom2):
        self.bondID = int(bondID)
        self.bondType = int(bondType)
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)
    
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return self.bondID == other.bondID
        else:
            return False
    def __neq__(self,other):
        return not self.__eq__(other)

class Angle(object):
    """The Angle object represents the angle between three connected atoms.  The format is the same as in the Angles section of a LAMMPS input file.
    
    Parameters
    ----------
    angleID : int
        The unique integer identifier of the angle, the same as the one defined in the LAMMPS inpit file
    angleType : int
        The integer identifier of the angle type which is the same as the angle type number defined in the LAMMPS input file
    atom1 : int
        The atom ID of the first atom associated with the angle, same as the one defined in LAMMPS input file.
    atom2 : int
        The atom ID of the second atom associated with the angle, same as the one defined in LAMMPS input file.
    atom3 : int
        The atom ID of the third atom associated with the angle, same as the one defined in LAMMPS input file.
    """
    def __init__(self, angleID,angleType,atom1,atom2,atom3):
        self.angleID = int(angleID)
        self.angleType =int(angleType)
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)
        self.atom3 = int(atom3)
        self.atoms = set([int(atom1),int(atom2),int(atom3)])
    
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return other.angleID == self.angleID
        else:
            return False
    def __neq__(self,other):
        return not self.__eq__(other)

class Dihedral(object):
    """The Dihedral object represents the dihedral between four connected atoms.

    Parameters
    ----------
    dihID : int
        The unique integer identifier for the dihedral same as the one in the LAMMPS input file.
    dihType : int
        The unique integer identifier for the dihedral type which corresponds to the dihedral type in the LAMMPS input file.
    atom1 : int
        The atom ID of the first atom in the dihedral.
    atom2 : int
        The atom ID of the second atom in the dihedral.
    atom3 : int
        The atom ID of the third atom in the dihedral.
    atom4 : int
        The atom ID of the fourth atom in the dihedral.
    """
    def __init__(self,dihID,dihType,atom1,atom2,atom3,atom4):
        self.dihID = int(dihID)
        self.dihType = int(dihType)
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)
        self.atom3 = int(atom3)
        self.atom4 = int(atom4)
        self.atoms = set([int(atom1),int(atom2),int(atom3),int(atom4)])
    
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return other.dihID == self.dihID
        else:
            return False
    def __neq__(self,other):
        return not self.__eq__(other)


def loadBonds(filename):
    """This function loads the Bonds from a LAMMPS input file and turns them into a list of Bond objects.
    
    Parameters
    ----------
    filename : str
        The filename of the LAMMPS input file which has the Bonds you wish to use.
    
    Returns
    -------
    Bond List
        A list of Bond objects with the same value as the Bonds in the LAMMPS input file passed in.
    """
    bonds = rdlmp.readBonds(filename)
    return [Bond(bond[0],bond[1],bond[2],bond[3]) for bond in bonds]

def loadAngles(filename):
    """This function loads the Angless from a LAMMPS input file and turns them into a list of Angle objects.
    
    Parameters
    ----------
    filename : str
        The filename of the LAMMPS input file which has the Angles you wish to use.
    
    Returns
    -------
    Angle List
        A list of Angle objects with the same value as the Angles in the LAMMPS input file passed in.
    """
    angles = rdlmp.readAngles(filename)
    return [Angle(angle[0],angle[1],angle[2],angle[3],angle[4]) for angle in angles]

def loadDihedrals(filename):
    """This function loads the Dihedrals from a LAMMPS input file and turns them into a list of Dihedral objects.
    
    Parameters
    ----------
    filename : str
        The filename of the LAMMPS input file which has the Dihedrals you wish to use.
    
    Returns
    -------
    Dihedral List
        A list of Dihedral objects with the same value as the Dihedrals in the LAMMPS input file passed in.
    """
    dihedrals = rdlmp.readDihedrals(filename)
    return [Dihedral(dihedral[0],dihedral[1],dihedral[2],dihedral[3],dihedral[4],dihedral[5]) for dihedral in dihedrals]

def getBondsFromAtoms(atoms,bonds):
    """Based on a list of atoms find all bonds in Bond list, bonds that contain these atoms.

    Parameters
    ----------
    atoms : Atom List
        A list of Atom objects which are the basis for searching through the bond list.
    bonds : Bond List
        A List of Bond objects which the function searches through.

    Returns
    -------
    Bond List
        A list of all Bond objects that contain any of the given Atom objects atoms.
    """
    id_combos = [(atom1.atomID,atom2.atomID) for (atom1,atom2) in permutations(atoms,r=2)]
    bondlist = []
    for bond in bonds:
        if (bond.atom1,bond.atom2) in id_combos:
            bondlist.append(bond)            
    return bondlist

def getAnglesFromAtoms(atomlist,bondlist,angles):
    """Given a list of Atoms find all the Angles in angles which contains these atoms.

    Parameters
    ----------
    atomlist : Atom List
        A list of Atoms used to search through the Angle List.
    bondlist : Bond List
        A List of Bond objects used to get connectivity of atoms.  The connectivity is used to find the possible angle combinations of the Atoms in atoms.
    angles : Angle List
        A List of Angles which is searched through.

    Returns
    -------
    anglelist : Angle List
        A list of Angles that are associated with the Atoms in atoms.
    """
    mol_graph = molecule2graph(atomlist,bondlist)
    angle_combos=[]
    #Get possible angles by getting subgraphs with only 2 steps
    for atom1,atom2 in permutations(mol_graph.__iter__(),r=2):
        if (ntwkx.shortest_path_length(mol_graph,source=atom1,target=atom2)==2):
            angle_combos.append(ntwkx.shortest_path(mol_graph,source=atom1,target=atom2))
    anglelist=[]
    for angle in angles:
        if([angle.atom1,angle.atom2,angle.atom3] in angle_combos):
            anglelist.append(angle)
    return anglelist

def getDihedralsFromAtoms(atomlist,bondlist,dihedrals):
    """Given a list of Atoms find all the Dihedral objects in dihedrals which contains these atoms.

    Parameters
    ----------
    atomlist : Atom List
        A list of Atoms used to search through the Dihedral List.
    bondlist : Bond List
        A List of Bond objects used to get connectivity of atoms.  The connectivity is used to find the possible dihedral combinations of the Atoms in atoms.
    angles : Angle List
        A List of Angles which is searched through.

    Returns
    -------
    anglelist : Angle List
        A list of Angles that are associated with the Atoms in atoms.
    """
    mol_graph = molecule2graph(atomlist,bondlist)
    dihedral_combos=[]
    #Get possible angles by getting subgraphs with only 2 steps
    for atom1,atom2 in permutations(mol_graph.__iter__(),r=2):
        if(ntwkx.shortest_path_length(mol_graph,source=atom1,target=atom2)==3):
            dihedral_combos.append(ntwkx.shortest_path(mol_graph,source=atom1,target=atom2))      
    dihedral_list=[]
    for dihedral in dihedrals:
        if([dihedral.atom1,dihedral.atom2,dihedral.atom3,dihedral.atom4] in dihedral_combos):
            dihedral_list.append(dihedral)
    return dihedral_list

def groupAtomsByMol(atoms):
    """Group atoms by their associated molecular ID
    
    Parameters
    ----------
    atoms : Atom List
        A list of Atom objects that you want grouped by molecule ID

    Returns
    -------
    mol_dict : Atom List Dictionary
        A dictionary with the molecule ID's as keys and the list of atoms associated with that molecule ID as the entries
    """
    mol_dict = {}
    for k,g in groupby(atoms,key=(lambda x: x.get_mol_ID())):
        mol_dict[k]=list(g)
    return mol_dict

def groupBondsByMol(mol_dict,bonds):
    """Group bonds by their associated molecular ID
    
    Parameters
    ----------
    mol_dict : Atom List Dictionary
        A dictionary of Atom List with each entry indexed by molecule ID and the entries with that index being an Atom list of Atom objects associated with the molecule ID
    bonds : Bond List
        A list of Bond objects that you want grouped by molecule ID

    Returns
    -------
    bond_dict : Bond List Dictionary
        A dictionary with the molecule ID's as keys and the list of bonds associated with that molecule ID as the entries
    """
    bond_dict={}
    for molid in mol_dict: 
        bond_dict[molid] = getBondsFromAtoms(mol_dict[molid],bonds)
    return bond_dict

def groupAnglesByMol(mol_dict,bond_dict,angles):
    """Group angles by their associated molecular ID
    
    Parameters
    ----------
    mol_dict : Atom List Dictionary
        A dictionary of Atom List with each entry indexed by molecule ID and the entries with that index being an Atom list of Atom objects associated with the molecule ID
    bond_dict : Bond List Dictionary
        A dictionary of Bond List items indexed by their associated molecule ID.
    angles : Angle List
        A list of Angle objects that you want grouped by molecule ID

    Returns
    -------
    angle_dict : Angle List Dictionary
        A dictionary with the molecule ID's as keys and the list of angles associated with that molecule ID as the entries
    """
    angle_dict={}
    for molid in mol_dict:
        angle_dict[molid]=getAnglesFromAtoms(mol_dict[molid],bond_dict[molid],angles)
    return angle_dict

def groupDihedralsByMol(mol_dict,bond_dict,dihedrals):
    """Group dihedrals by theri associated molecule ID

    Parameters
    ----------
    mol_dict : Atom List Dictionary
        A dictionary of Atom Lists indexed by their associated molecule ID's
    bond_dict : Bond List Dictionary
        A dictionary of Bond Lists indexed by their associated molecule ID's
    dihedrals : Dihedral List
        A list of dihedrals to by grouped by molecule ID.

    Returns
    -------
    dihedral_dict : Dihedral List Dictionary
        A dictionary of Dihedral Lists indexed by their associated molecule ID.
    """
    dihedral_dict={}
    for molID in mol_dict:
        dihedral_dict[molID]=getDihedralsFromAtoms(mol_dict[molID],bond_dict[molID],dihedrals)
    return dihedral_dict

def set_anchor_atoms(molecules,anchortype):
    """For every molecule in molecules set the anchot atom to the atom with atom type anchortype.

    Parameters
    ----------
    molecules : list of Molecule
    A list of Molecule objects that will have their anchor atoms set to the atom with type anchortype.
    anchortype : int
    The atom type of the atom to set as the anchor for each Molecule in molecules.
    """
    for key, molecule in molecules.items():
        anchorIDs = [atom.atomID for atom in molecule.atoms if atom.atomType==anchortype]
        if len(anchorIDs)>0:
            molecule.setAnchorAtom(anchorIDs[0])

def molecule2graph(atomlist,bondlist):
    """Converts the Atom list and Bond list of a molecule to a graph data object

    Parameters
    ----------
    atomlist : Atom List
        A list of atoms in the molecule
    bondlist : Bond List
        A list of bonds associated with the molecule

    Returns
    -------
    molecule_graph : Networkx Graph
        A Networkx Graph object with the nodes being the atom ID's of the atoms and the connectivity of the graph defined by the bonds in the bondlist.
    """
    molecule_graph = ntwkx.Graph()
    for atom in atomlist:
        molecule_graph.add_node(atom.atomID)
    molecule_graph.add_edges_from([(bond.atom1,bond.atom2) for bond in bondlist])
    return molecule_graph

def constructMolecules(filename):
    """From a LAMMPS input file construct a list of Molecule objects based on the molecules in the LAMMPS input file.

    Parameters
    ----------
    filename : str
        The name of the LAMMPS input file that contains the molecules

    Returns
    -------
    molecules : Molecule List
        A list of Molecule objects with the data specified by the LAMMPS input file passed in.
    """
    print("Loading Data File:")
    atoms = atm.loadAtoms(filename)
    bonds = loadBonds(filename)
    angles = loadAngles(filename)
    dihedrals = loadDihedrals(filename)

    print("Grouping Atoms by Molecule")
    atom_dict = groupAtomsByMol(atoms)
    print("Grouping Bonds by Molecule")
    bond_dict = groupBondsByMol(atom_dict,bonds)
    print("Grouping Angles by Molecule")
    angle_dict = groupAnglesByMol(atom_dict,bond_dict,angles)
    print("Grouping Dihedrals by ID")
    dihedral_dict = groupDihedralsByMol(atom_dict,bond_dict,dihedrals)
    print("Assembling Molecules")
    molecules = {}
    for molID in atom_dict:
        molecules[molID]=Molecule(molID,atom_dict[molID],bond_dict[molID],angle_dict[molID],dihedral_dict[molID])
    return molecules


def rot_quat(vector,theta,rot_axis):
    """Rotates a vector about a specified axis a specified angle theta using the quaternion method

    Parameters
    ----------
    vector : float vector
        A vector of three elements that represents the X, Y, Z coordinates of the vector that one wishes to rotate
    theta : float
        The angle in radians by which the vector rotates.
    rot_axis : float vector
        A vector of three elements which represents the X,Y,Z elements of the rotation axis.

    Returns
    -------
    new_vector : float vector
        A vector of three elements representing the X,Y,Z coordinates of the old vector after rotation
    """
    if np.linalg.norm(rot_axis)==0.:
        raise ValueError("The rotation axis must have a non-zero magnitude in order for rotation about the axis to make sense.")
    rot_axis = rot_axis/np.linalg.norm(rot_axis)
    vector_mag = np.linalg.norm(vector)
    quat = np.array([cos(theta/2),sin(theta/2)*rot_axis[0],sin(theta/2)*rot_axis[1],sin(theta/2)*rot_axis[2]])
    quat_inverse = np.array([cos(theta/2),-sin(theta/2)*rot_axis[0],-sin(theta/2)*rot_axis[1],-sin(theta/2)*rot_axis[2]])

    vect_quat = np.array([0,vector[0],vector[1],vector[2]])/vector_mag
    new_vector = quat_mult(quat_mult(quat,vect_quat),quat_inverse)
    return new_vector[1:]*vector_mag

def quat_mult(q1,q2):
    """Matrix multiplication for two 1x4 matrices.
    
    Parameters
    ----------
    q1,q2 : Numpy arrays of floats
        1x4 arrays to be multiplied.
    
    Returns
    -------
    results : Numpy array of floats
        A 1x4 array containing the matrix product of q1 and q2.
    """
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    w = w1*w2-x1*x2-y1*y2-z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    result = np.array([w,x,y,z])
    return result

def getAngle(atom1,atom2,atom3):
    """Returns the smallest angle defined by three atoms.
    
    Parameters
    ----------
    atom1,atom2,atom3 : Atom
        The atom objects which define the desired angle; atom2 is the central atom. 
        
    Returns
    -------
    angle : float
        The angle between the vector from atom2 to atom1 and the vector from atom2 to atom3.
    """
    line1 = atom1.position-atom2.position
    line2 = atom3.position-atom2.position
    angle = np.arccos(np.dot(line1,line2) / (np.linalg.norm(line1)*np.linalg.norm(line2)))
    return angle




















