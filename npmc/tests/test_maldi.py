import os,sys
#sys.path.insert(0,os.path.abspath('../src'))

import numpy as np
import networkx as ntwkx
import unittest
import pickle
from math import *
import sys
import npmc.maldi_class as mldi
import npmc.molecule_class as mlc

script_path = os.path.dirname(os.path.realpath(__file__))

class TestMALDI(unittest.TestCase):

    def setUp(self):
        self.data_file = os.path.abspath(script_path+"/test_files/maldi_tests/nanoparticle/system.data")
        self.anchor_type = 4
        self.numsamples = 1000
        self.nn_distance = 8.0
        self.graph_index = 0
        self.ligands_per_fragment = 2
        self.maldi = mldi.MALDISpectrum(self.data_file,self.anchor_type,self.numsamples,self.nn_distance,self.graph_index,self.ligands_per_fragment,(13,5))

    def test_get_random_molecule_returns_molecule(self):
        self.assertIsInstance(self.maldi.get_random_molecule(),mlc.Molecule,msg="get_random_molecule does not return an object of type Molecule")

    def test_get_maldi_spectrum_returns_histogram_with_correct_size(self):
        self.assertEqual(len(self.maldi.get_maldi_spectrum()[0]),self.maldi.ligands_per_fragment+1,msg = "Histogram returned by get_maldi_spectrum is not the expected size.")
    
    def test_molecules_graph(self):
        import pdb; pdb.set_trace()
