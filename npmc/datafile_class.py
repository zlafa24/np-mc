import numpy as np
import npmc.atom_class as atm
import npmc.molecule_class as mlc
from npmc.nanoparticle_class import Nanoparticle as nanop

class XYZAtom(object):
    
    def __init__(self,elem,x,y,z):
        self.elem = elem
        self.position = np.array([x,y,z])


class XYZFile(object):
    """A class encapsulating an XYZ file, also containing built in functions for adding predefined molecules (currently just alkanethiols).
    """
    def __init__(self,filename):
        self.file =filename
        self.elements = np.empty((0,4))

    def create_alkanethiol(self,chainlength):
    """A functioned defined for adding an alkanethiol with a given chainlength to the XYZFile object.

    Parameters
    ----------
    chainlength : int
        Chainlength of alkanethiol to be added to the XYZFile object.
    """
        self.elements = np.empty((chainlength,4),dtype=np.float)
        self.elements[0,:]=[32,0.,0.,0.]
        current_pos = np.array([0.,0.,0.])
        for i,element in enumerate(self.elements[1:chainlength,:]):
            current_pos+=np.array([1.018,0.,0.])
            current_pos[1]=0.562 if i%2==0 else 0.
            element[:] = np.concatenate(([12],current_pos))

    def create_hollow_np(self,outer_radius,inner_radius):
        """Creates a hollow icosahedral nanoparticle of defined radius and adds it to the XYZFile object.

        Parameters
        ----------
        outer_radius : float
            Outer radius of hollow icosahedron to be added to the XYZFile object
        
        inner_radius : float
            Inner radius of hollow icosahedron to be added to the XYZFile object
        """
        np_object = nanop.create_hollow_icosahedron(outer_radius,inner_radius)
        atoms = np_object.atoms
        np_elements = atoms_to_xyz(atoms)
        self.elements = np.vstack((self.elements,np_elements))

    def write_to_file(self):
        """Writes the elements contained within the XYZFile object to an XYZ file with the filename given by the instance variable self.file.
        """
        numelements = self.elements.shape[0]
        np.savetxt(self.file,self.elements,header="{}\n".format(numelements),fmt = '%i %5.4f %5.4f %5.4f',comments="")



class TemplateSystem(object):
    """A class encapsulating a system lt file with specified molecule lt files.
    """
    def __init__(self,tmols,quantities):
        self.lt_mols = tmols
        self.nmol1, self.nmol2 = quantities

    def write_system_to_lt(self):
        with open('system.lt','w') as sysfile:
            for lt_mol in self.lt_mols:
                sysfile.write(lt_mole.file)
            


class TemplateMolecule(object):
    """A class encapsulating a molecule lt file.
    """
    def __init__(self,filename : str,molecule : mlc.Molecule):
        self.file = filename
        self.mol_name = filename.split(".")[0]
        self.molecule = molecule

    @classmethod
    def from_alkanethiol(cls,chainlength,filename):
        return(cls(molecule=cls.create_alkanethiol(chainlength),
            filename=filename))

    def write_to_lt(self):
        with open(self.file,'w') as lt_file:
             lt_file.write("{} {{\n".format(self.mol_name.upper()))
             self.write_atoms_to_lt(lt_file)
             self.write_bonds_to_lt(lt_file)
             lt_file.write("}")
    
    @classmethod
    def create_alkanethiol(cls,chainlength):
        atoms = np.empty(chainlength+1,dtype=object)
        bonds = np.empty(chainlength,dtype=object)
        atoms[0]= atm.Atom(0,1,32)
        current_pos = np.array([0.,0.,0.])
        bonds[:] = [mlc.Bond(i,1,i,i+1) for i in range(chainlength)]
        bonds[0].bondType=0
        for i in range(1,chainlength):
            current_pos+=np.array([1.018,0.,0.])
            current_pos[1]=0.5602 if i%2==0 else 0.
            atoms[i]=atm.Atom(i,1,14,position=current_pos)
        atoms[-1]=atm.Atom(i,1,15,position=current_pos)
        return(mlc.Molecule(1,atoms,bonds,angles=None,dihedrals=None))
 
    def write_atoms_to_lt(self,lt_file):
        lt_file.write("\twrite(\"Data Atoms\"){\n")
        for atom in self.molecule.atoms:
            lt_file.write(self.atom_to_string(atom))
        lt_file.write("\t}\n")

    def write_bonds_to_lt(self,lt_file):
        lt_file.write("\twrite(\"Data Bonds\"){\n")
        for bond in self.molecule.bonds:
            lt_file.write(self.bond_to_string(bond))
        lt_file.write("\t}\n")


    def atom_to_string(self,atom):
        atomtype_dict = {32:'S',14:'C2',15:'C3'}
        return("\t\t$atom:{0:d}\t$mol:.\t@atom:{1}\t{2:4.3f}\t{3:5.3f}\t{4:5.3f}\t{5:5.3f}\n".format(atom.atomID,
                                                                       atomtype_dict[atom.atomType],
                                                                       atom.charge,
                                                                       atom.position[0],atom.position[1],atom.position[2]))
    def bond_to_string(self,bond):
        bondtype_dict = {0:'SC',1:'CC'}
        return("\t\t$bond:{0:d}\t@bond:{1}\t$atom:{2:d}\t$atom:{3:d}\n".format(bond.bondID,bondtype_dict[bond.bondType],bond.atom1,bond.atom2))
        

def atoms_to_xyz(atoms):
    elements = np.empty((len(atoms),4))
    for i,atom in enumerate(atoms):
        elements[i,:]=np.concatenate(([atom.atomType],atom.position))
    return(elements)


def xyzfile_to_atoms(xyzfile):
    atoms=np.empty(len(xyzfile.elements),dtype=object)
    for i,element in enumerate(xyzfile.elements):
        atoms[i] = atm.Atom(i,1,element[0],position=element[1:4])
    return(atoms)

def xyzfile_to_molecule(xyzfile):
    atoms = xyzfile_to_atoms(xyzfile)
    return(mlc.Molecule(1,atoms,bonds=None,angles=None,dihedrals=None))
