import numpy as np
import npmc.atom_class as atm
import npmc.molecule_class as mlc
from npmc.nanoparticle_class import Nanoparticle as nanop

class XYZAtom(object):
    
    def __init__(self,elem,x,y,z):
        self.elem = elem
        self.position = np.array([x,y,z])


class XYZFile(object):
    
    def __init__(self,filename):
        self.file =filename
        self.elements = np.empty((0,4))

    def create_alkanethiol(self,chainlength):
        self.elements = np.empty((chainlength,4),dtype=np.float)
        self.elements[0,:]=[32,0.,0.,0.]
        current_pos = np.array([0.,0.,0.])
        for i,element in enumerate(self.elements[1:chainlength,:]):
            current_pos+=np.array([1.018,0.,0.])
            current_pos[1]=0.562 if i%2==0 else 0.
            element[:] = np.concatenate(([12],current_pos))

    def create_hollow_np(self,radius):
        np_object = nanop.create_hollow_icosahedron(radius,10)
        atoms = np_object.atoms
        np_elements = atoms_to_xyz(atoms)
        self.elements = np.vstack((self.elements,np_elements))

    def write_to_file(self):
        numelements = self.elements.shape[0]
        np.savetxt(self.file,self.elements,header="{}\n".format(numelements),fmt = '%i %5.4f %5.4f %5.4f',comments="")



class TemplateMolecule(object):

    def __init__(self,molecule,filename):
        self.molecule = molecule
        self.file = filename
        self.mol_name = filename.split(".")[0]

    def write_to_lt(self):
        with open(self.file,'w') as lt_file:
             lt_file.write("{} {{\n".format(self.mol_name.upper()))
             self.write_atoms_to_lt(lt_file)
             self.write_bonds_to_lt(lt_file)
             lt_file.write("}")
 
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
        return("\t\t$atom:{0:d}\t$mol:.\t@atom:{1:d}\t{2:4.3f}\t{3:5.3f}\t{4:5.3f}\t{5:5.3f}\n".format(atom.atomID,
                                                                       atom.atomType,
                                                                       atom.charge,
                                                                       atom.position[0],atom.position[1],atom.position[2]))
    def bond_to_string(self,bond):
        return("\t\t$bond:{0:d}\t@bond:{1:d}\t$atom:{2:d}\t$atom:{3:d}\n".format(bond.bondID,bond.bondType,bond.atom1,bond.atom2))
        

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
