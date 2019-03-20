import sys, os
import unittest
#sys.path.insert(0,os.path.abspath('../src'))

from npmc.atom_class import Atom, loadAtoms

script_path = os.path.dirname(os.path.realpath(__file__))

class TestAtomMethods(unittest.TestCase):
    def setUp(self):
        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.test_file = os.path.abspath(script_path+'/test_files/meoh.data') 
        self.atom = Atom(1,2,3,1.1,[4,6,7])
        self.atom_list = [Atom(1,1,1,1.1,[0.474,0.000,1.059]),
                Atom(2,1,1,0.000,[-0.621, 0.0, -0.007]),
                Atom(3,1,3,0.265,[-0.125, 0.0, 2.357]),
                Atom(4,1,2,-0.700,[0.138, 0.0, -1.654]),
                Atom(5,1,4,0.435,[0.598, 0.0, 2.999])]

    def test_atom_equality_operator(self):
        self.assertEqual(self.atom == Atom(1,2,3,1.1,[4,6,7]),True,msg="Atom.__eq__ does not return True when two atoms with the same atom ID are compared")
    def test_atom_equality_operator_when_not_the_same(self):
        self.assertEqual(self.atom == Atom(3,2,3,1.1,[4,6,7]),False,msg="Atom.__eq__ does not return False when two atoms with the different atom ID are compared")
    def test_get_atom_ID(self):
        self.assertEqual(self.atom.get_atom_ID(),1,msg="get_atom_ID method does not return correct ID")
    def test_get_mol_ID(self):
        self.assertEqual(self.atom.get_mol_ID(),2,msg="get_mol_ID method does not return correct ID")
    def test_get_type(self):
        self.assertEqual(self.atom.get_type(),3,msg="get_type method does not return correct atom type")
    def test_get_charge(self):
        self.assertEqual(self.atom.get_charge(),1.1,msg="get_charge method does not return correct atom charge")
    def test_get_position(self):
        self.assertEqual(self.atom.get_pos(),[4,6,7],msg="get_pos returns wrong position")
    def test_loadAtoms(self):
        self.assertSequenceEqual(loadAtoms(self.test_file),self.atom_list,msg="loadAtoms return doesn't match expected Atom objects")
