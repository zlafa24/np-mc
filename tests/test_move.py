import os,sys
sys.path.insert(0,os.path.abspath('../src'))

import simulation_class as sim
import numpy as np
import unittest
import pickle
from math import *
import sys
import move_class as mvc
import molecule_class

script_path = os.path.abspath(".")

class TestCBMCRegrowth(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        self.lt_directory = os.path.abspath('./test_files/move_tests/lt_files/two_meohs')
        self.dihedral_type3_pdf =  pickle.load(open('./test_files/move_tests/dihedral_type3_pdf.pickle','rb'))
        self.dihedral_type4_pdf = pickle.load(open('./test_files/move_tests/dihedral_type4_pdf.pickle','rb'))
        self.SCCO_pair_pdf = pickle.load(open('./test_files/move_tests/pair_pe_pdf_SCCO_dihedral_angle.pickle','rb'))
        self.init_file = os.path.abspath(self.lt_directory+'/system.in')
        self.data_file = os.path.abspath(self.lt_directory+'/system.data')
        self.dump_file = os.path.abspath(self.lt_directory+'/regrow.xyz')
        self.temp = 298.15
        self.simulation = sim.Simulation(init_file=self.init_file,datafile=self.data_file,dumpfile=self.dump_file,temp=self.temp)
        self.cbmc_move = mvc.CBMCRegrowth(self.simulation,2)

    def test_select_random_molecule_returns_molecule(self):
        self.assertIsInstance(self.cbmc_move.select_random_molecule(),molecule_class.Molecule,msg="select_random_molecule does not return an object of type Molecule.")

    def test_select_index_returns_value_within_correct_range(self):
        test_molecule = self.cbmc_move.select_random_molecule()
        index = self.cbmc_move.select_index(test_molecule)
        self.assertTrue(3<=index<len(test_molecule.atoms),msg="Index returned by select_index is outside the bounds of the prescribed range.")

    def test_select_dih_angles_returns_correct_pdf_after_1000000_trials_for_a_CCOH_OPLS_dihedral(self):
        cbmc_move_large_trials = mvc.CBMCRegrowth(self.simulation,2,numtrials=10000000)
        (normed_histogram,bins) = np.histogram(cbmc_move_large_trials.select_dih_angles(4),bins=500,density=True)
        np.testing.assert_array_almost_equal(normed_histogram,self.dihedral_type4_pdf,decimal=2,err_msg="The resulting histogram from 100000 trials of select_dih_angles does not match the distriburion expected by the PDF of the OPLS dihedral type for a CCOH dihedral.")

    def test_evaluate_energies_returns_expected_energies_for_specified_angles(self):
        molecule = self.simulation.molecules[1]
        rotations = [0,pi,2*pi,pi/2.,2*pi]
        energies = self.cbmc_move.evaluate_energies(molecule,4,rotations)
        actual_energies = [-1.1082622,-1.34260189,-1.1082622,-1.55146693,-1.1082622]
        np.testing.assert_array_almost_equal(energies,actual_energies,err_msg="evaluate_energies does not return correct energies for a set of specified rotation angles.")

    def test_turn_off_molecule_atoms_for_2_MeOH_system_returns_correct_energy_after_turning_off_hydrogen(self):
        self.cbmc_move.turn_off_molecule_atoms(self.cbmc_move.simulation.molecules[1],3)
        self.assertAlmostEqual(-1.6710847+1.1598538,self.cbmc_move.simulation.get_pair_PE(),places=5,msg="Energy obtained after turning off hydrogen in 2 MeOH system using turn_off_molecule_atoms is not the expected value.")
    
    def test_turn_off_molecule_atoms_for_2_MeOH_system_returns_correct_energy_after_turning_off_hydrogen_and_oxygen(self):
        self.cbmc_move.turn_off_molecule_atoms(self.cbmc_move.simulation.molecules[1],2)
        self.assertAlmostEqual(-1.4003423+0.0120187,self.cbmc_move.simulation.get_pair_PE(),places=5,msg="Energy obtained after turning off hydrogen in 2 MeOH system using turn_off_molecule_atoms is not the expected value.")
 

    def test_regrow_MeOH_from_index_3_in_2_MeOH_system(self):
        self.cbmc_move.regrow(self.simulation.molecules[1],3)

    def tearDown(self):
        os.chdir(script_path)
        self.cbmc_move.simulation.turn_on_all_atoms()



class TestTranslationMove(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        self.lt_directory = os.path.abspath('./test_files/move_tests/lt_files/two_meohs')
        self.init_file = os.path.abspath(self.lt_directory+'/system.in')
        self.data_file = os.path.abspath(self.lt_directory+'/system.data')
        self.dump_file = os.path.abspath(self.lt_directory+'/regrow.xyz')
        self.temp = 298.15
        self.simulation = sim.Simulation(init_file=self.init_file,datafile=self.data_file,dumpfile=self.dump_file,temp=self.temp)
        self.translate_move = mvc.TranslationMove(self.simulation,0.5)

    def test_translate_translates_molecule_by_specified_move(self):
        molecule = self.simulation.molecules[1]
        old_positions = np.copy([atom.position for atom in molecule.atoms])
        move = np.array([1,1,1])
        self.translate_move.translate(molecule,move)
        self.simulation.get_coords()
        molecule = self.simulation.molecules[1]
        new_positions = [atom.position for atom in molecule.atoms]
        np.testing.assert_allclose(new_positions,old_positions+move,err_msg="translate method of Translate class does not translate molecule as specified by passed in move.")

    def tearDown(self):
        os.chdir(script_path)

