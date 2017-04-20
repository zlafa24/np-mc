import os,sys
sys.path.insert(0,os.path.abspath('../src'))

from simulation_class import *
import numpy as np
import unittest
import pickle
from math import *
import sys

script_path = os.path.abspath(".")

class TestSimulationInitializations(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.current_folder = os.path.abspath(".")
        self.data_folder =  os.path.abspath("./test_files/simulation_tests/lt_files/nanoparticle_1ddt_1meoh/")
        self.init_file = self.data_folder+"/system.in"
        self.data_file = self.data_folder+"/system.data"
        self.dumpfile =  self.data_folder+"/regrow.xyz"
        self.silver_expected_coords = np.loadtxt(self.current_folder+"/test_files/simulation_tests/expected_silver.xyz",skiprows=2)
        self.adsorbate_expected_coords =  np.loadtxt(self.current_folder+"/test_files/simulation_tests/expected_adsorbate.xyz",skiprows=2)
        self.temp = 298.15
        self.sim = Simulation(self.init_file,self.data_file,self.dumpfile,self.temp)

    def test_initialize_group_by_comparing_silver_xyz_file_with_expected_file(self):
        self.sim.dump_group("silver","silverdump.xyz") 
        actual_coords = np.loadtxt(self.data_folder+"/silverdump.xyz",skiprows=2)
        np.testing.assert_almost_equal(actual_coords,self.silver_expected_coords,decimal=4,err_msg="Coords do not match")
    
    def test_initialize_group_by_comparing_adsorbate_xyz_file_with_expected_file(self):
        self.sim.dump_group("adsorbate","adsorbate.xyz") 
        actual_coords = np.loadtxt(self.data_folder+"/adsorbate.xyz",skiprows=2)
        np.testing.assert_almost_equal(actual_coords,self.adsorbate_expected_coords,decimal=4,err_msg="Coords do not match")

class TestSimulationPotentialEvaluationsIntramolecular(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.current_folder = script_path
        self.data_folder =  os.path.abspath(script_path+"/test_files/simulation_tests/lt_files/meoh/")
        self.init_file = self.data_folder+"/system.in"
        self.data_file = self.data_folder+"/system.data"
        self.dumpfile =  self.data_folder+"/regrow.xyz"
        self.temp = 298.15
        self.sim = Simulation(self.init_file,self.data_file,self.dumpfile,self.temp)

    def test_getCoulPE_with_lone_MeOH(self):
        #import pdb;pdb.set_trace()
        self.assertAlmostEqual(0.0000,self.sim.getCoulPE(),places=4,msg="getCoulPE does not return the expected coulombic energy for a lone MeOH (it should be 0)")

    def test_getVdwlPE_with_lone_MeOH(self):
        self.assertAlmostEqual(-0.14060,self.sim.getVdwlPE(),places=4,msg="getCoulPE does not return the expected Van der Waals energy for a lone MeOH")


class TestSimulationPotentialEvaluationIntermolecular(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.current_folder = script_path
        self.data_folder =  os.path.abspath(script_path+"/test_files/simulation_tests/lt_files/two_meohs/")
        self.init_file = self.data_folder+"/system.in"
        self.data_file = self.data_folder+"/system.data"
        self.dumpfile =  self.data_folder+"/regrow.xyz"
        self.temp = 298.15
        self.sim = Simulation(self.init_file,self.data_file,self.dumpfile,self.temp)

    def test_getVdwlPE_with_two_MeOHs_separated_by_5A(self):
        self.assertAlmostEqual(-1.6710847,self.sim.getVdwlPE(),places=4,msg="getVdwlPE does not return expected value for Van der Waals energy of two MeOHs seoarated by 5Angstroms")

    def test_getCoulPE_with_two_MeOHs_separated_by_5A(self):
        self.assertAlmostEqual(0.5628225,self.sim.getCoulPE(),places=4,msg="getCoulPE does not return expected value for Coulombic energy of two MeOHs separated by 5A")
