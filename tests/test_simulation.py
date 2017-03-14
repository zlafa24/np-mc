import os,sys
sys.path.insert(0,os.path.abspath('../src'))

from simulation_class import *
import numpy as np
import unittest
import pickle
from math import *

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


"""
if __name__ == "__main__":
    data_folder =  os.path.abspath("./test_files/simulation_tests/lt_files/nanoparticle_1ddt_1meoh/")
    init_file = data_folder+"/system.in"
    data_file = data_folder+"/system.data"
    dumpfile =  data_folder+"/regrow.xyz"
    expected_coords = np.loadtxt("./test_files/simulation_tests/expected_silver.xyz",skiprows=2)
    temp = 298.15

    sim = Simulation(init_file,data_file,dumpfile,temp)

    sim.dump_group("silver","silverdump.xyz")
    
    actual_coords = np.loadtxt(data_folder+"/silverdump.xyz",skiprows=2)
    
    np.testing.assert_almost_equal(actual_coords,expected_coords,err_msg="Coords do not match")
    #sim.minimize()
"""
