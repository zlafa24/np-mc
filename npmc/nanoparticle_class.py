"""
This module contains classes and functions used to create nanoparticles of arbitrary shapes
"""
import npmc.atom_class as atmc
from scipy.spatial import Delaunay
import os
import numpy as np
from scipy.constants import golden_ratio
from itertools import product
from math import sqrt

class Nanoparticle(object):
    """
    Nanoparticle class used to encapsulate a nanoparticle atoms and write to xyz. Nanoparticle is created by carving out shape from a list of atoms passed in.
    Currently nanoparticles are carved out from an initial FCC cubic lattice.  To create an hollow icosahedral XYZ file use:
    >>> import npmc.nanoparticle_class as npc
    >>> ico_np = npc.Nanoparticle.create_hollow_icosahedron(outer_radius=15,inner_radius=10)
    >>> ico_no.write_to_xyz('nanoparticle.xyz')
    """
    def __init__(self,atoms):
        self.atoms = atoms

    @classmethod
    def create_icosahedron(cls,radius):
        phi = golden_ratio
        unit_vertices = np.array([[0,1,phi],
                            [0,-1,phi],
                            [0,1,-phi],
                            [0,-1,-phi],
                            [1,phi,0],
                            [-1,phi,0],
                            [1,-phi,0],
                            [-1,-phi,0],
                            [phi,0,1],
                            [-phi,0,1],
                            [phi,0,-1],
                            [-phi,0,-1]])
        scaled_vertices = radius*unit_vertices
        hull = Delaunay(scaled_vertices)
        atoms = Nanoparticle.get_atoms_inside(hull) 
        return(cls(atoms))

    @classmethod
    def create_hollow_icosahedron(cls,outer_radius,inner_radius,lattice_units=20,lattice_const=4.08):
        phi = golden_ratio
        unit_vertices = np.array([[0,1,phi],
                            [0,-1,phi],
                            [0,1,-phi],
                            [0,-1,-phi],
                            [1,phi,0],
                            [-1,phi,0],
                            [1,-phi,0],
                            [-1,-phi,0],
                            [phi,0,1],
                            [-phi,0,1],
                            [phi,0,-1],
                            [-phi,0,-1]])
        outer_vertices = outer_radius*unit_vertices
        outer_hull = Delaunay(outer_vertices)
        atoms1 = Nanoparticle.get_atoms_inside(outer_hull,lattice_units,lattice_const)
        ids1 = set([atom.atomID for atom in atoms1])
        inner_vertices = inner_radius*unit_vertices
        inner_hull = Delaunay(inner_vertices)
        atoms2 = Nanoparticle.get_atoms_outside(inner_hull,lattice_units,lattice_const)
        ids2 = set([atom.atomID for atom in atoms2])
        ids = ids1.intersection(ids2)
        atoms = [atom for atom in atoms1 if atom.atomID in list(ids)]
        return(cls(atoms))

    @classmethod
    def create_tetrahedron(cls,outer_edge_length,lattice_units=20,lattice_const=4.08):
        unit_vertices = np.array([[1,0,-(1/sqrt(2))],
                                  [-1,0,-(1/sqrt(2))],
                                  [0,1,1/sqrt(2)],
                                  [0,-1,1/sqrt(2)]])*0.5
        outer_vertices = outer_edge_length*unit_vertices
        outer_hull = Delaunay(outer_vertices)
        atoms = Nanoparticle.get_atoms_inside(outer_hull,lattice_units,lattice_const)
        return(cls(atoms))


    @classmethod
    def create_hollow_tetrahedron(cls,outer_edge_length,inner_edge_length):
        unit_vertices = np.array([[1,0,-(1/sqrt(2))],
                                  [-1,0,-(1/sqrt(2))],
                                  [0,1,1/sqrt(2)],
                                  [0,-1,1/sqrt(2)]])*0.5
        outer_vertices = outer_edge_length*unit_vertices
        inner_vertices = inner_edge_length*unit_vertices
        outer_hull = Delaunay(outer_vertices)
        inner_hull = Delaunay(inner_vertices)
        atoms1 = Nanoparticle.get_atoms_outside(inner_hull)
        ids1 = set([atom.atomID for atom in atoms1])
        atoms2 = Nanoparticle.get_atoms_inside(outer_hull)
        ids2 = set([atom.atomID for atom in atoms2])
        ids = ids1.intersection(ids2)
        atoms = [atom for atom in atoms1 if atom.atomID in list(ids)]
        return(cls(atoms))

    @staticmethod
    def create_fcc_lattice(lattice_units,lattice_const,atom_type=1):
        ls = range(1,lattice_units+1)
        ms = range(1,lattice_units+1)
        ns = range(1,lattice_units+1)
        
        numatoms = 4*len(list(product(ls,ms,ns)))
        basis_atoms = lattice_const*np.array([[0.,0.,0.],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0.]])
        positions=np.empty((numatoms,4))
        atoms = np.empty((numatoms),dtype=object)
        for lattice_pos,(l,m,n) in enumerate(product(ls,ms,ns)):
            current_position=np.array([l,m,n])*lattice_const-10*lattice_const*np.array([1,1,1])
            for i,atom in enumerate(basis_atoms):
                #position[i]=np.array((atom_type,atom+current_position))
                atoms[lattice_pos*4+i]=atmc.Atom(lattice_pos*4+i,1,atomType=atom_type,position=atom+current_position)
        return(atoms)


    @staticmethod
    def is_point_inside(hull,point):
        return(hull.find_simplex(point)>=0)
    
    @staticmethod
    def get_atoms_outside(hull,lattice_units=20,lattice_const=4.08):
        #template_atoms = atmc.loadAtoms(os.path.abspath("../lt_files/nanoparticle_template/lts/nanoparticle.data"))
        template_atoms = Nanoparticle.create_fcc_lattice(lattice_units,lattice_const)
        atoms=[]
        for atom in template_atoms:
            if not Nanoparticle.is_point_inside(hull,atom.position):
                atoms.append(atom)
        return(atoms)

    @staticmethod
    def get_atoms_inside(hull,lattice_units=20,lattice_const=4.08):  
        template_atoms = Nanoparticle.create_fcc_lattice(lattice_units,lattice_const)
        #template_atoms = atmc.loadAtoms(os.path.abspath("../lt_files/nanoparticle_template/lts/nanoparticle.data"))
        atoms = []
        for atom in template_atoms:
            if Nanoparticle.is_point_inside(hull,atom.position):
                atoms.append(atom)
        return(atoms)

    def get_min_max_difference(self):
        coords = np.array([atom.position for atom in self.atoms])
        max_x = np.max(coords[:,0])
        min_x = np.min(coords[:,0])
        return(max_x-min_x)

    def print_xyz_file(self,filename='nanoparticle.xyz'):
        positions = np.array([np.insert(atom.position,0,1) for atom in self.atoms])
        np.savetxt(filename,positions,fmt='%d %.4f %.4f %.4f',header=str(len(self.atoms))+'\n',comments="")
