import read_lmp_rev6 as rdlmp

class Atom(object):
	def __init__(self,atomID,molID,atomType,charge=0.,position=[0.,0.,0.]):
		self.atomID = int(atomID)
		self.molID = int(molID)
		self.atomType = int(atomType)
		self.charge = float(charge)
		self.position = position
	
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
	atoms = rdlmp.readAtoms(filename)
	return [Atom(atom[0],atom[1],atom[2],atom[3],atom[4:7]) for atom in atoms]
