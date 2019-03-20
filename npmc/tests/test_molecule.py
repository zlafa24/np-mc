import sys, os
import unittest
import pickle
from math import *
#sys.path.insert(0,os.path.abspath('../src'))

from npmc.molecule_class import *

script_dir = os.path.dirname(os.path.realpath(__file__))

def plotDihedral(molecule,dihedral):
        positions = [molecule.getAtomByID(atomID).position for atomID in (dihedral.atom1,dihedral.atom2,dihedral.atom3,dihedral.atom4)]
        print(positions)

class TestMoleculeLoadFunctions(unittest.TestCase):
    def setUp(self):
        self.bond = Bond(1,1,1,4)
        self.meoh_file = os.path.abspath(script_dir+"/test_files/meoh.data")
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
        self.three_meoh_file = os.path.abspath(script_dir+'/test_files/molecule_tests/three_meoh.data')
        self.atoms = pickle.load(open(script_dir+'/test_files/molecule_tests/atoms.pickle','rb'))
        self.bonds = pickle.load(open(script_dir+'/test_files/molecule_tests/bonds.pickle','rb'))
        self.angles = pickle.load(open(script_dir+'/test_files/molecule_tests/angles.pickle','rb'))
        self.dihedrals = pickle.load(open(script_dir+'/test_files/molecule_tests/dihedrals.pickle','rb'))
        self.atom_dict = pickle.load(open(script_dir+'/test_files/molecule_tests/atom_dict.pickle','rb'))
        self.bond_dict = pickle.load(open(script_dir+'/test_files/molecule_tests/bond_dict.pickle','rb'))
        self.angle_dict = pickle.load(open(script_dir+'/test_files/molecule_tests/angle_dict.pickle','rb'))
        self.dih_dict = pickle.load(open(script_dir+'/test_files/molecule_tests/dih_dict.pickle','rb'))
        self.mol_dict = pickle.load(open(script_dir+'/test_files/molecule_tests/molecule_dict.pickle','rb'))

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
        self.longMessage=True
        self.theta1 = pi/3.
        self.axis1 = [0,cos(pi/3),sin(pi/3)]
        self.vector1 = [1,-1,2]
        self.theta2 = pi/4
        self.axis2 = [0,0,1] 
        self.vector2 = [0,1,0]
        self.theta3=pi/2
        self.axis3=[-0.5066642,0.0,-0.4932]
        self.vector3=[-0.629,0,2.364]

    """Based on worked example by the Institute of Mathematical Sciences (http://www.imsc.res.in/~knr/131129workshop/writeup_knr.pdf)"""
    def test_rot_quat_returns_expected_result_from_IMS_Example1(self):
        np.testing.assert_almost_equal(rot_quat(self.vector1,self.theta1,self.axis1),[(10.+4*sqrt(3))/8.,(1+2*sqrt(3))/8.,(14-3*sqrt(3))/8.],decimal=3,err_msg="rot_quat produced a differemt vector then a proven worked example")

    def test_rot_quat_returns_expected_result_from_rotating_y_unit_vector_about_z_axis(self):
        np.testing.assert_almost_equal(rot_quat(self.vector2,self.theta2,self.axis2),[-1./sqrt(2),1./sqrt(2),0],decimal=3,err_msg="rot_quat produced a differemt vector then the expected result of rotating yhat about the z axis 45 degrees")

    def test_rot_quat_returns_expected_value_when_rotates_dihedral_pi_2_degrees(self):
        np.testing.assert_almost_equal(rot_quat(self.vector3,self.theta3,self.axis3),[0.8586349,2.1326423,0.8358948],decimal=3,err_msg="rot_quat failed to provide the right coordinates from rotating example dihedral 90 degrees clockwise")

class TestMoleculeClassMethods(unittest.TestCase):
    
    def assertAlmostEqualAngles(self,theta1,theta2,places,msg):
        angle_diff = atan2(sin(theta2-theta1),cos(theta2-theta1))
        self.assertAlmostEqual(angle_diff,0.0,places=places,msg=msg)

    def setUp(self):
        self.longMessage = True
        self.mol_dict = pickle.load(open(script_dir+'/test_files/molecule_tests/molecule_dict.pickle','rb'))
        self.anchorAtom = self.mol_dict[1].atoms[0]
        self.atom1 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom1)
        self.atom2 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom2)
        self.atom3 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom3)
        self.atom4 = self.mol_dict[1].getAtomByID(self.mol_dict[1].dihedrals[0].atom4)

    def test_setAnchorAtom_returns_false_when_given_incorrect_atomID(self):
        self.assertEqual(self.mol_dict[1].setAnchorAtom(20),False,msg="setAnchorAtom didn't return False when given atomID not in molecule")

    def test_setAnchorAtom_sets_anchor_atom_when_given_correct_atomID(self):
        self.assertEqual(self.mol_dict[1].setAnchorAtom(1),True,msg = "setAnchorAtom didn't return True when given correct atomID")
        self.assertEqual(self.mol_dict[1].anchorAtom,self.anchorAtom,msg="Incorrect atom is set as anchor Atom")

    def test_getDihedralAngle_when_atoms_are_trans_in_same_plane_returns_pi(self):
        self.assertAlmostEqual(self.mol_dict[1].getDihedralAngle(self.mol_dict[1].dihedrals[0]),pi,places=3,msg="getDihedralAngle does not return pi even though atoms are all in the same plane")

    def test_getDihedralAngle_returns_expected_result_when_dihedral_rotated_with_rot_quat_pi_div_2_radians(self):
        vector = self.atom4.position - self.atom3.position
        axis = self.atom3.position-self.atom2.position
        theta = -pi/2
        self.atom4.position = rot_quat(vector,theta,axis)+self.atom3.position
        self.assertAlmostEqual(self.mol_dict[1].getDihedralAngle(self.mol_dict[1].dihedrals[0]),pi/2,places=2,msg = "Rotation of dihedral using rot_quat to rotate dihedral 180 degrees counterclockwise does not produce correct result when calculated with getDihedralAngle")

    def test_getDihedralAngle_returns_expected_result_when_dihedral_rotated_with_rot_quat_plus_pi_radians(self):
        vector = self.atom4.position - self.atom3.position
        axis = self.atom3.position-self.atom2.position
        theta = pi
        self.atom4.position = rot_quat(vector,theta,axis)+self.atom3.position
        self.assertAlmostEqualAngles(self.mol_dict[1].getDihedralAngle(self.mol_dict[1].dihedrals[0]) , 0 , places=2 , msg = "Rotation of dihedral using rot_quat to rotate dihedral 180 degreees clockwise does not produce correct result when calculated with getDihedralAngle")
    
    def test_getAtomByMolIndex_returns_correct_atom_when_passed_valid_index(self):
        self.mol_dict[1].setAnchorAtom(1)
        self.assertEqual(self.mol_dict[1].getAtomByMolIndex(3),self.atom4,msg="getAtomByMolIndex does not return correct atom when given a valid index")

    def test_getAtomsByMolIndex_returns_None_when_passed_index_out_of_bounds(self):
        self.assertIsNone(self.mol_dict[1].getAtomByMolIndex(5),msg="getAtomsByMolIndex allowed an out of range index to be used without returning None")

    def test_rotateDihedral_raises_exception_when_index_less_than_three(self):
        with self.assertRaises(ValueError,msg="rotateDihedral did not raise an exception when the index was less than 3"):
            self.mol_dict[1].rotateDihedral(2,pi/2)

    def test_rotateDihedral_raises_exception_when_index_greater_then_atom_length_minus_one(self):
        with self.assertRaises(ValueError,msg="rotateDihedral did not raise an exception when the index passed was greater than the atom list length minus one"):
            self.mol_dict[1].rotateDihedral(5,pi/2)

    def test_rotateDihedral_correctly_rotates_test_MeOH_and_returns_correct_coords_after_pi_div_2_rotation(self):
        self.mol_dict[1].setAnchorAtom(1)
        self.mol_dict[1].rotateDihedral(3,pi/2)
        actual_coords = np.array([self.mol_dict[1].getAtomByMolIndex(0).position,
                                self.mol_dict[1].getAtomByMolIndex(1).position,
                                self.mol_dict[1].getAtomByMolIndex(2).position,
                                self.mol_dict[1].getAtomByMolIndex(3).position,
                                self.mol_dict[1].getAtomByMolIndex(4).position,])
        correct_coords = np.array([[0.138, 0.,-1.654],
                                    [0.474,0.,1.059],
                                    [-0.621,0.,-0.007],
                                    [0.8152293,1.34789,1.3911922],
                                    [1.5073144,1.303574,2.064948117]])
        np.set_printoptions(threshold=np.inf)
        np.testing.assert_almost_equal(actual_coords,correct_coords,decimal=3,err_msg="rotateDihedral does not return correct coords after rotating MeOH by first dihedral pi/2 radians",verbose=True)
                                
    def test_align_to_vector_returns_correct_coords_when_aligning_MeOH_to_z_unit_vector(self):
        molecule = self.mol_dict[1]
        molecule.setAnchorAtom(1)
        old_vector = np.copy(molecule.get_com()-molecule.anchorAtom.position)
        vector = np.array([0,-1,0])
        molecule.align_to_vector(vector)
        new_vector = np.copy(molecule.get_com()-molecule.anchorAtom.position)
        new_unit_vector = new_vector/np.linalg.norm(new_vector)
        np.testing.assert_allclose(new_unit_vector,vector,atol=1e-10,err_msg="align_to_vector does not properly align molecule to a given vector")
        

    def test_get_com_returns_expected_center_of_mass(self):
        expected_com = np.array([0.0928,0.0,0.9508])
        molecule = self.mol_dict[1]
        actual_com = molecule.get_com()
        np.testing.assert_allclose(actual_com,expected_com,atol=1e-10,err_msg="get_com doesnt return expected center of mass for a MeOH molecule.")


    def test_move_atoms_correctly_shifts_molecule(self):
        molecule = self.mol_dict[1]
        old_positions = np.copy(np.array([atom.position for atom in molecule.atoms]))
        molecule.move_atoms(np.array([5,5,5]))
        new_positions = np.array([atom.position for atom in molecule.atoms])
        np.testing.assert_allclose(old_positions+np.array([5,5,5]),new_positions,err_msg="move_atoms does not correctly shift atoms by specified amount/direction.")

    def test_move_atoms_by_index_correctly_shifts_molecule(self):
        molecule = self.mol_dict[1]
        molecule.setAnchorAtom(1)
        old_positions = np.copy(np.array([molecule.getAtomByMolIndex(i).position for i in range(len(molecule.atoms))]))
        molecule.move_atoms_by_index(np.array([5,5,5]),2)
        new_positions = np.array([molecule.getAtomByMolIndex(i).position for i in range(len(molecule.atoms))])
        old_positions[2:,:]+=np.array([5,5,5])
        np.testing.assert_allclose(old_positions,new_positions,err_msg="move_atoms does not correctly shift atoms by specified amount/direction.")















