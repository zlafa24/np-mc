"""This module contains the Atom class and all the helper functions associated with atoms.

"""
import npmc.read_lmp_rev6 as rdlmp
class Atom(object):
    """The Atom class represents an atom described by the LAMMPS full atom style.

    Parameters
    ----------
    atomID : int
        The unique integer identifier of the atom as specified in the LAMMPS input file.
    molID : int
        The unique integer identifier of the molecule associated with this atom as specified in the LAMMPS input file.
    atomType : int
        The integer that identifies the atom type as specified in the LAMMPS input file.
    charge : float, optional
        The charge on the atom defaults to 0.
    position : float vector, optional
        A three element vector which represent the X,Y,Z coordinates of the atom.  It defaults to a vector of [X=0,Y=0,Z=0].
    """
    def __init__(self,atomID,molID,atomType,charge=0.,position=[0.,0.,0.]):
        self.atomID = int(atomID)
        self.molID = int(molID)
        self.atomType = int(atomType)
        self.charge = float(charge)
        self.position = position
    
    def __eq__(self,other):
        if isinstance(other, self.__class__):
            return self.atomID == other.atomID
        else:
            return False
    def __neq__(self,other):
        return not self.__eq__(other)
    def get_pos(self):
        return self.position
    def get_charge(self):
        return self.charge
    def get_type(self):
        return self.atomType
    def get_mol_ID(self):
        return self.molID
    def get_atom_ID(self):
        return self.atomID

def loadAtoms(filename):
    """Loads the atoms from a LAMMPS  input file and returns a list of Atom object which represent those atoms.
    
    Parameters
    ----------
    filename : str
        The name of the LAMMPS input file which contain the atoms

    Returns
    -------
    Atom List
        A list of Atom objects which contains the atom info in the given LAMMPS input file.
    """
    atoms = rdlmp.readAtoms(filename)
    atom_list = [Atom(atom[0],atom[1],atom[2],atom[3],atom[4:7]) for atom in atoms]
    return atom_list






