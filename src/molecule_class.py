import read_lmp_rev6 as rdlmp
import atom_class as atm
from itertools import groupby, permutations
import networkx as ntwkx

class Molecule(object):
	def __init__(self,molID,atoms,bonds,angles,dihedrals):
		self.molID = molID
		self.atoms = atoms
		self.bonds = bonds
		self.angles = angles
		self.dihedrals = dihedrals
	def setAnchorAtom(self,atomID):
		atom = self.getAtomByID(atomID)
		if(atom!=None):
			self.anchorAtom = atom
			return True
		else:
			return False
	
    def atomsAsGraph(self):
        return molecule2graph(self.atoms,self.bonds)
    def getAtomByID(self,atomID):
		for atom in self.atoms:
			if(atom.atomID==atomID):
				return atom
		return None

class Bond(object):
	def __init__(self,bondID,bondType,atom1,atom2):
		self.bondID = int(bondID)
		self.bondType = int(bondType)
		self.atom1 = int(atom1)
		self.atom2 = int(atom2)

class Angle(object):
	def __init__(self, angleID,angleType,atom1,atom2,atom3):
		self.angleID = int(angleID)
		self.angleType =int(angleType)
		self.atom1 = int(atom1)
		self.atom2 = int(atom2)
		self.atom3 = int(atom3)

class Dihedral(object):
	def __init__(self,dihID,dihType,atom1,atom2,atom3,atom4):
		self.dihID = int(dihID)
		self.dihType = int(dihType)
		self.atom1 = int(atom1)
		self.atom2 = int(atom2)
		self.atom3 = int(atom3)
		self.atom4 = int(atom4)
	def getAngle(self):
		b1 = self.atom2.position-self.atom1.position
		b2 = self.atom3.position-self.atom2.position
		b3 = self.atom4.position-self.atom3.position
		b2norm = b2/np.linalg.norm(b2)
		n1 = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
		n2 = np.cross(b2,b3)/np.linalg.norm(np.cross(b2,b3))
		m1 = np.cross(n1,b2norm)
		angle = atan2(np.dot(m1,n2),np.dot(n1,n2))
		angle=((angle-pi)*(-1)+2*pi)%(2*pi)
		return angle
	def rotate(self,angle):
		rot_axis = self.atom3.position-self.atom2.position
		rot_angle = angle-self.getAngle()
		#for atom in atoms2rotate:
		#	atom[4:7] = rot_quat((atom[4:7]-dih_atoms[2,4:7]),rot_angle,rot_axis)+dih_atoms[2,4:7]



def loadBonds(filename):
	bonds = rdlmp.readBonds(filename)
	return [Bond(bond[0],bond[1],bond[2],bond[3]) for bond in bonds]

def loadAngles(filename):
	angles = rdlmp.readAngles(filename)
	return [Angle(angle[0],angle[1],angle[2],angle[3],angle[4]) for angle in angles]

def loadDihedrals(filename):
	dihedrals = rdlmp.readDihedrals(filename)
	return [Dihedral(dihedral[0],dihedral[1],dihedral[2],dihedral[3],dihedral[4],dihedral[5]) for dihedral in dihedrals]

def getBondsFromAtoms(atoms,bonds):
	id_combos = [(atom1.atomID,atom2.atomID) for (atom1,atom2) in permutations(atoms,r=2)]
	bondlist = []
	for bond in bonds:
		if (bond.atom1,bond.atom2) in id_combos:
			bondlist.append(bond)			
	return bondlist

def getAnglesFromAtoms(atomlist,bondlist,angles):
	mol_graph = molecule2graph(atomlist,bondlist)
	angle_combos=[]
	#Get possible angles by getting subgraphs with only 2 steps
	for atom1,atom2 in permutations(mol_graph.nodes_iter(),r=2):
		if(ntwkx.shortest_path_length(mol_graph,source=atom1,target=atom2)==2):
				angle_combos.append(ntwkx.shortest_path(mol_graph,source=atom1,target=atom2))	  
	anglelist=[]
	for angle in angles:
		if([angle.atom1,angle.atom2,angle.atom3] in angle_combos):
			anglelist.append(angle)
	return anglelist

def getDihedralsFromAtoms(atomlist,bondlist,dihedrals):
	mol_graph = molecule2graph(atomlist,bondlist)
	dihedral_combos=[]
	#Get possible angles by getting subgraphs with only 2 steps
	for atom1,atom2 in permutations(mol_graph.nodes_iter(),r=2):
		if(ntwkx.shortest_path_length(mol_graph,source=atom1,target=atom2)==3):
				dihedral_combos.append(ntwkx.shortest_path(mol_graph,source=atom1,target=atom2))	  
	dihedral_list=[]
	for dihedral in dihedrals:
		if([dihedral.atom1,dihedral.atom2,dihedral.atom3,dihedral.atom4] in dihedral_combos):
			dihedral_list.append(dihedral)
	return dihedral_list

def groupAtomsByMol(atoms):
	mol_dict = {}
	for k,g in groupby(atoms,key=(lambda x: x.get_mol_ID())):
		mol_dict[k]=list(g)
	return mol_dict

def groupBondsByMol(mol_dict,bonds):
	bond_dict={}
	for molid in mol_dict: 
		bond_dict[molid] = getBondsFromAtoms(mol_dict[molid],bonds)
	return bond_dict

def groupAnglesByMol(mol_dict,bond_dict,angles):
	angle_dict={}
	for molid in mol_dict:
		angle_dict[molid]=getAnglesFromAtoms(mol_dict[molid],bond_dict[molid],angles)
	return angle_dict

def groupDihedralsByMol(mol_dict,bond_dict,dihedrals):
	dihedral_dict={}
	for molID in mol_dict:
		dihedral_dict[molID]=getDihedralsFromAtoms(mol_dict[molID],bond_dict[molID],dihedrals)
	return dihedral_dict

def molecule2graph(atomlist,bondlist):
	molecule_graph = ntwkx.Graph()
	for atom in atomlist:
		molecule_graph.add_node(atom.atomID)
	molecule_graph.add_edges_from([(bond.atom1,bond.atom2) for bond in bondlist])
	return molecule_graph

def constructMolecules(filename):
	print "Loading Data File:"
	atoms = atm.loadAtoms(filename)
	bonds = loadBonds(filename)
	angles = loadAngles(filename)
	dihedrals = loadDihedrals(filename)

	print "Grouping Atoms by Molecule"
	atom_dict = groupAtomsByMol(atoms)
	print "Grouping Bonds by Molecule"
	bond_dict = groupBondsByMol(atom_dict,bonds)
	print "Grouping Angles by Molecule"
	angle_dict = groupAnglesByMol(atom_dict,bond_dict,angles)
	print "Grouping Dihedrals by ID"
	dihedral_dict = groupDihedralsByMol(atom_dict,bond_dict,dihedrals)
	print "Assembling Molecules" 
	molecules = {}
	for molID in atom_dict:
		molecules[molID]=Molecule(molID,atom_dict[molID],bond_dict[molID],angle_dict[molID],dihedral_dict[molID])
	return molecules


def rot_quat(vector,theta,rot_axis):
	rot_axis = rot_axis/np.linalg.norm(rot_axis)
	vector_mag = np.linalg.norm(vector)
	quat = np.array([cos(theta/2),sin(theta/2)*rot_axis[0],sin(theta/2)*rot_axis[1],sin(theta/2)*rot_axis[2]])
	quat_inverse = np.array([cos(theta/2),-sin(theta/2)*rot_axis[0],-sin(theta/2)*rot_axis[1],-sin(theta/2)*rot_axis[2]])
	quat = quat/np.linalg.norm(quat)
	quat_inverse = quat_inverse/(np.linalg.norm(quat_inverse)**2)

	vect_quat = np.array([0,vector[0],vector[1],vector[2]])/vector_mag
	new_vector = quat_mult(quat_mult(quat,vect_quat),quat_inverse)
	return new_vector[1:]*vector_mag


