import sys, os
import unittest
import pickle
import matplotlib.pyplot as plt
from math import *
sys.path.insert(0,os.path.abspath('../src'))

from molecule_class import *

def plotDihedral(molecule,dihedral):
        positions = [molecule.getAtomByID(atomID).position for atomID in (dihedral.atom1,dihedral.atom2,dihedral.atom3,dihedral.atom4)]
        print positions


class TestMoleculeLoadFunctions(unittest.TestCase):
    def setUp(self):
        self.bond = Bond(1,1,1,4)
        self.meoh_file = os.path.abspath("./test_files/meoh.data")
        self.bond_list = [Bond(1,1,1,4),
                        Bond(2,2,1,2),
                        Bond(3,3,2,3),
                        Bond(4,4,3,5)]
        self.angle_list = [Angle(1,1,2,1,4),
                        Angle(2,2,1,2,3),
                        Angle(3,3,2,3,5)]
        self.dihedral_list = [Dihedral(1,1,3,2,1,4),
                            Dihedral(2,2,1,2,3,5)]

    def test_loadBonds(self):
        self.assertSequenceEqual(loadBonds(self.meoh_file),self.bond_list,msg="Bond list from loadBonds does not match expected Bond list")

    def test_loadAngles(self):
        self.assertSequenceEqual(loadAngles(self.meoh_file),self.angle_list,msg="Angle List from loadAngles does not match expected Angle List")

    def test_loadDihedrals(self):
        self.assertSequenceEqual(loadDihedrals(self.meoh_file),self.dihedral_list,msg = "Dihedral List from loadDihedrals does not match expected Dihedral List")
    

class TestMoleculeGroupByFunctions(unittest.TestCase):
    def setUp(self): 
        self.three_meoh_file = os.path.abspath("./test_files/molecule_tests/three_meoh.data")
        self.atoms = pickle.load(open('./test_files/molecule_tests/atoms.pickle','rb'))
        self.bonds = pickle.load(open('./test_files/molecule_tests/bonds.pickle','rb'))
        self.angles = pickle.load(open('./test_files/molecule_tests/angles.pickle','rb'))
        self.dihedrals = pickle.load(open('./test_files/molecule_tests/dihedrals.pickle','rb'))
        self.atom_dict = pickle.load(open('./test_files/molecule_tests/atom_dict.pickle','rb'))
        self.bond_dict = pickle.load(open('./test_files/molecule_tests/bond_dict.pickle','rb'))
        self.angle_dict = pickle.load(open('./test_files/molecule_tests/angle_dict.pickle','rb'))
        self.dih_dict = pickle.load(open('./test_files/molecule_tests/dih_dict.pickle','rb'))
        self.mol_dict = pickle.load(open('./test_files/molecule_tests/molecule_dict.pickle','rb'))

    def test_getBondsFromAtoms_returns_expected_output(self):
        self.assertSequenceEqual(getBondsFromAtoms(self.atom_dict[2],self.bonds),self.bond_dict[2],msg="Bond List obtained by getBondsFromAtoms does not correspond with the expected output")

    def test_getAnglesFromAtoms_returns_expected_output(self):
        self.assertSequenceEqual(getAnglesFromAtoms(self.atom_dict[2],self.bond_dict[2],self.angles),self.angle_dict[2],msg="Angle List obtained by getAnglesFromAtoms does not correspond to expected to ouput")

    def test_getDihedralsFromAtoms_returns_expected_output(self):
        self.assertSequenceEqual(getDihedralsFromAtoms(self.atom_dict[2],self.bond_dict[2],self.dihedrals),self.dih_dict[2],msg="Dihedral List obtained by getDihedralsFromAtoms does not correspond to expected output")

    def test_groupAtomsByMol_returns_expected_output(self):
        self.assertDictEqual(groupAtomsByMol(self.atoms),self.atom_dict,msg="Dictionary of Atom List constructed by groupAtomsByMol does not match expected Atom List Dictionary")

    def test_groupBondsByMol_returns_expected_output(self):
        self.assertDictEqual(groupBondsByMol(self.atom_dict,self.bonds),self.bond_dict,msg="Bond List Dictionary produced by groupBondsByMol doesn't match correct results")

    def test_groupAnglesByMol_returns_expected_output(self):
        self.assertDictEqual(groupAnglesByMol(self.atom_dict,self.bond_dict,self.angles),self.angle_dict,msg="Angle List Dictionary from groupAnglesByMol returns different result from correct result")

    def test_groupDihedralsByMol_returns_expected_output(self):
        self.assertDictEqual(groupDihedralsByMol(self.atom_dict,self.bond_dict,self.dihedrals),self.dih_dict,msg="Dihedral List Dictionary from groupDihedralsByMol returns different result from the correct one.")

    def test_constructMolecule_correctly_constructs_molecule_from_3_meoh(self):
        self.assertDictEqual(constructMolecules(self.three_meoh_file),self.mol_dict,msg="constructMolecule doesn't produce correct Molecule Dictionary")

class TestQuaternionRotation(unittest.TestCase):
    def setUp(self):
        self.theta1 = pi/3.
        self.axis1 = [0,cos(pi/3),sin(pi/3)]
        self.vector1 = [1,-1,2]
        self.theta2 = pi/3
        self.axis2 = [1,1,1] 
        self.vector2 = [0,0,1]

    """Based on worked example by the Institute of Mathematical Sciences (http://www.imsc.res.in/~knr/131129workshop/writeup_knr.pdf)"""
    def test_rot_quat_returns_expected_result_from_IMS_Example1(self):
        np.testing.assert_almost_equal(rot_quat(self.vector1,self.theta1,self.axis1),[(10.+4*sqrt(3))/8.,(1+2*sqrt(3))/8.,(14-3*sqrt(3))/8.],decimal=3,err_msg="rot_quat produced a differemt vector then a proven worked example")

    #def test_rot_quat_returns_expected_result_from_IMS_Example1(self):



class TestMoleculeClassMethods(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        self.mol_dict = pickle.load(open('./test_files/molecule_tests/molecule_dict.pickle','rb'))
        self.anchorAtom = self.mol_dict[1].atoms[0]
        self.atom1 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom1)
        self.atom2 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom2)
        self.atom3 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom3)
        self.atom4 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom4)

    def test_setAnchorAtom_returns_false_when_given_incorrect_atomID(self):
        self.assertEquals(self.mol_dict[1].setAnchorAtom(20),False,msg="setAnchorAtom didn't return False when given atomID not in molecule")

    def test_setAnchorAtom_sets_anchor_atom_when_given_correct_atomID(self):
        self.assertEqual(self.mol_dict[1].setAnchorAtom(1),True,msg = "setAnchorAtom didn't return True when given correct atomID")
        self.assertEqual(self.mol_dict[1].anchorAtom,self.anchorAtom,msg="Incorrect atom is set as anchor Atom")

    def test_getDihedralAngle_when_atoms_are_trans_in_same_plane_returns_pi(self):
        self.assertAlmostEqual(self.mol_dict[1].getDihedralAngle(self.mol_dict[1].dihedrals[0]),pi,places=3,msg="getDihedralAngle does not return pi even though atoms are all in the same plane")

    def test_getDihedralAngle_returns_expected_result_when_dihedral_rotated_with_rot_quat(self):
        vector = self.atom4.position - self.atom3.position
        axis = self.atom3.position-self.atom2.position
        theta = -3*pi/2
        self.atom4.position = rot_quat(vector,theta,axis)
        self.assertAlmostEqual(self.mol_dict[1].getDihedralAngle(self.mol_dict[1].dihedrals[0]),3*pi/2,places=2,msg = "Rotation of dihedral using rot_quat does not produce correct result when calculated with getDihedralAngle")












