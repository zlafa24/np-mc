import numpy as np
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

    def __init__(self,molecule):
        self.molecule = molecule

    def write_to_lt(self):


    def write_atom(self):
        

def atoms_to_xyz(atoms):
    elements = np.empty((len(atoms),4))
    for i,atom in enumerate(atoms):
        elements[i,:]=np.concatenate(([atom.atomType],atom.position))
    return(elements)


