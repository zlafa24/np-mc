import os,sys
#sys.path.insert(0,os.path.abspath('../src'))

import npmc.simulation_class as sim
import numpy as np
import networkx as ntwkx
import unittest
import mock
import pickle
from math import *
import sys
import npmc.move_class as mvc
import npmc.molecule_class as mlc
import npmc.forcefield_class as ffc
import pdb

script_path = os.path.dirname(os.path.realpath(__file__))

class TestCBMCRegrowth(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        self.lt_directory = os.path.abspath(script_path+'/test_files/move_tests/lt_files')
        self.dihedral_type3_pdf =  pickle.load(open(script_path+'/test_files/move_tests/dihedral_type3_pdf.pickle','rb'))
        self.dihedral_type4_pdf = pickle.load(open(script_path+'/test_files/move_tests/dihedral_type4_pdf.pickle','rb'))
        self.SCCO_pair_pdf = pickle.load(open(script_path+'/test_files/move_tests/pair_pe_pdf_SCCO_dihedral_angle.pickle','rb'))
        self.init_file = os.path.abspath(self.lt_directory+'/two_meohs/system.in')
        self.data_file = os.path.abspath(self.lt_directory+'/two_meohs/system.data')
        self.dump_file = os.path.abspath(self.lt_directory+'/two_meohs/regrow.xyz')
        self.init_file_b = os.path.abspath(self.lt_directory+'/hexo/hexo.in')
        self.data_file_b = os.path.abspath(self.lt_directory+'/hexo/hexo.data')
        self.dump_file_b = os.path.abspath(self.lt_directory+'/hexo/hexo.xyz')
        self.temp = 298.15
        self.simulation = sim.Simulation(init_file=self.init_file,datafile=self.data_file,dumpfile=self.dump_file,temp=self.temp)
        self.sim_branched = sim.Simulation(init_file=self.init_file_b,datafile=self.data_file_b,dumpfile=self.dump_file_b,temp=self.temp)
        self.cbmc_move = mvc.CBMCRegrowth(self.simulation,2,(5,5))
        self.cbmc_move_b = mvc.CBMCRegrowth(self.sim_branched,5,(9,9))

    def test_select_random_molecule_returns_molecule(self):
        self.assertIsInstance(self.cbmc_move.select_random_molecule(),mlc.Molecule,msg="select_random_molecule does not return an object of type Molecule.")

    def test_select_index_returns_value_within_correct_range(self):
        test_molecule = self.cbmc_move.select_random_molecule()
        index = self.cbmc_move.select_index(test_molecule)
        self.assertTrue(3<=index<len(test_molecule.atoms),msg="Index returned by select_index is outside the bounds of the prescribed range.")

    def test_select_dih_angles_returns_correct_pdf_after_1000000_trials_for_a_CCOH_OPLS_dihedral(self):
        cbmc_move_large_trials = mvc.CBMCRegrowth(self.simulation,2,(5,5),numtrials=10000000)
        molecule = cbmc_move_large_trials.molecules[2]
        dihedrals,atoms = molecule.index2dihedrals(4)
        (normed_histogram,bins) = np.histogram(cbmc_move_large_trials.select_dih_angles([dihedrals]),bins=500,density=True)
        np.testing.assert_array_almost_equal(normed_histogram,self.dihedral_type4_pdf,decimal=2,err_msg="The resulting histogram from 100000 trials of select_dih_angles does not match the distriburion expected by the PDF of the OPLS dihedral type for a CCOH dihedral.")

    def test_evaluate_energies_returns_expected_energies_for_specified_angles(self):
        molecule = self.simulation.molecules[1]
        rotations = [0,pi,2*pi,pi/2.,2*pi]
        energies = self.cbmc_move.evaluate_energies(molecule,[molecule.getAtomByMolIndex(4)],rotations)
        actual_energies = [-1.1082622,-1.34260189,-1.1082622,-1.55146693,-1.1082622]
        np.testing.assert_array_almost_equal(energies,actual_energies,err_msg="evaluate_energies does not return correct energies for a set of specified rotation angles.")
        
    def test_internal_bond_angle_energy_calculation_matches_LAMMPS(self):
        lammps = self.simulation.lmp.extract_compute("ang_pe",0,0)
        internal = 0
        for molID,mol in self.simulation.molecules.items():
            for angle in mol.angles:
                force_field = [ff for ff in self.cbmc_move.angle_ffs if ff.angle_type==angle.angleType][0]
                internal += force_field.ff_function(getAngles([mol.getAtomByID(angle.atom1),mol.getAtomByID(angle.atom2),mol.getAtomByID(angle.atom3)],1)[0])
        np.testing.assert_array_almost_equal(lammps,internal,err_msg="LAMMPS and internal bond angle energy calculations are not equal.")
    
    def test_angle_calculation_from_two_dihedral_angles(self):
        mol = self.sim_branched.molecules[1460]
        predecessor_dict = dict(ntwkx.bfs_predecessors(mol.graph,source=mol.anchorAtom.atomID))
        successor_dict = dict(ntwkx.bfs_successors(mol.graph,source=mol.anchorAtom.atomID))
        for node in mol.graph:
            if len(mol.graph[node])>2: 
                branch_atoms = set(list(mol.graph[node].keys())+[node]+[predecessor_dict[predecessor_dict[node]]])
                dihedrals = [dihedral for dihedral in mol.dihedrals if dihedral.atoms.issubset(branch_atoms)]
                angle_atomIDs = list(mol.graph[node].keys())+[node]; angle_atomIDs.remove(predecessor_dict[node])
                bond_angle1 = getAngles([mol.getAtomByID(predecessor_dict[node]),mol.getAtomByID(node),mol.getAtomByID(successor_dict[node][0])],1)
                bond_angle2 = getAngles([mol.getAtomByID(predecessor_dict[node]),mol.getAtomByID(node),mol.getAtomByID(successor_dict[node][1])],1)
                break
        dihedral_angle1 = mol.getDihedralAngle(dihedrals[0])
        dihedral_angle2 = mol.getDihedralAngle(dihedrals[1])
        bond_angle = getAngles([mol.getAtomByID(angle_atomIDs[0]),mol.getAtomByID(angle_atomIDs[2]),mol.getAtomByID(angle_atomIDs[1])],1)
        bond_angle_calc = ffc.central_angle_Vincenty(dihedral_angle1,dihedral_angle2,bond_angle1,bond_angle2)
        np.testing.assert_array_almost_equal(bond_angle,bond_angle_calc,err_msg="The bond angle is incorrectly calculated from two dihedral angles.")

    def test_turn_off_molecule_atoms_for_2_MeOH_system_returns_correct_energy_after_turning_off_hydrogen(self):
        self.cbmc_move.turn_off_molecule_atoms(self.cbmc_move.simulation.molecules[1],3)
        self.assertAlmostEqual(-1.6710847+1.1598538,self.cbmc_move.simulation.get_pair_PE(),places=5,msg="Energy obtained after turning off hydrogen in 2 MeOH system using turn_off_molecule_atoms is not the expected value.")
      
    def test_turn_off_molecule_atoms_for_2_MeOH_system_returns_correct_energy_after_turning_off_hydrogen_and_oxygen(self):
        self.cbmc_move.turn_off_molecule_atoms(self.cbmc_move.simulation.molecules[1],2)
        self.assertAlmostEqual(-1.4003423+0.0120187,self.cbmc_move.simulation.get_pair_PE(),places=5,msg="Energy obtained after turning off hydrogen in 2 MeOH system using turn_off_molecule_atoms is not the expected value.")

    @mock.patch.object(mvc.CBMCRegrowth,'evaluate_energies')
    def test_evaluate_trial_rotations_raises_exception_when_probs_do_not_sum_to_1(self,mock_method):
        mock_method.return_value = np.array([1e20,1e20,1e20,1e20,1e20])
        self.assertRaises(ValueError,self.cbmc_move.evaluate_trial_rotations,self.simulation.molecules[1],4,5)

    @mock.patch.object(mvc.CBMCRegrowth,'evaluate_trial_rotations')
    def test_regrow_returns_False_when_evaluate_trial_rotations_raises_ValueError(self,mock_method):
        mock_method.side_effect = ValueError('Probabilities do not sum to 1')
        self.assertFalse(self.cbmc_move.regrow(self.simulation.molecules[1],4),msg="regrow doesn't return False when evaluate_trial_rotations raises a ValueError exception")

    @mock.patch.object(mvc.CBMCRegrowth,'regrow')
    def test_move_returns_False_when_regrow_returns_False(self,mock_method):
        mock_method.return_value = False
        self.assertFalse(self.cbmc_move.move(),msg="CBMCRegrowth move method does not return False when regrow method returns False")

    def test_regrow_MeOH_from_index_3_in_2_MeOH_system(self):
        self.cbmc_move.regrow(self.simulation.molecules[1],3)

    def tearDown(self):
        os.chdir(script_path)
        self.cbmc_move.simulation.turn_on_all_atoms()



class TestTranslationMove(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        self.lt_directory = os.path.abspath(script_path+'/test_files/move_tests/lt_files/two_meohs')
        self.init_file = os.path.abspath(self.lt_directory+'/system.in')
        self.data_file = os.path.abspath(self.lt_directory+'/system.data')
        self.dump_file = os.path.abspath(self.lt_directory+'/regrow.xyz')
        self.temp = 298.15
        self.simulation = sim.Simulation(init_file=self.init_file,datafile=self.data_file,dumpfile=self.dump_file,temp=self.temp)
        self.translate_move = mvc.TranslationMove(self.simulation,0.5,[])

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

class TestCBMCSwap(unittest.TestCase):
    def setUp(self):
        self.longMessage = True
        self.lt_directory = os.path.abspath(script_path+'/test_files/move_tests/lt_files/nanoparticle_1ddt_1meoh')
        self.init_file = os.path.abspath(self.lt_directory+'/system.in')
        self.data_file = os.path.abspath(self.lt_directory+'/system.data')
        self.dump_file = os.path.abspath(self.lt_directory+'/regrow.xyz')
        self.temp = 298.15
        self.simulation = sim.Simulation(init_file=self.init_file,datafile=self.data_file,dumpfile=self.dump_file,temp=self.temp)
        self.swap_move = mvc.CBMCSwap(self.simulation,anchortype=4,type_lengths=(5,13))

    def test_cbmcswap_is_child_of_cbmcregrowth(self):
        self.assertTrue(issubclass(mvc.CBMCSwap,mvc.CBMCRegrowth))

    def test_select_random_molecules_returns_molecules_of_correct_types(self):
        molecule1,molecule2=self.swap_move.select_random_molecules()
        self.assertEqual(len(molecule1.atoms),self.swap_move.type1_numatoms)
        
    def test_swap_anchor_positions_correctly_swaps_locations_of_anchors(self):
        molecule1 = self.simulation.molecules[2446]
        molecule2 = self.simulation.molecules[2447]
        position1_old = np.copy([atom.position for atom in molecule1.atoms])
        self.swap_move.swap_anchor_positions(molecule1,molecule2)
        self.simulation.get_coords()
        position2_new = np.copy([atom.position for atom in molecule2.atoms])
        np.testing.assert_allclose(position1_old[0,:],position2_new[0,:])

    def test_align_mol_to_positions_correctly_aligns_atom_positions(self):
        molecule1,molecule2=self.swap_move.select_random_molecules()
        position1_old = np.copy(np.array([molecule1.getAtomByMolIndex(i).position for i in range(len(molecule1.atoms))]))
        position2_old =  np.copy(np.array([molecule2.getAtomByMolIndex(i).position for i in range(len(molecule2.atoms))]))
        self.swap_move.align_mol_to_positions(molecule1,position2_old[0:(self.swap_move.starting_index+1)])
        position2_new = np.array([molecule2.getAtomByMolIndex(i).position for i in range(len(molecule2.atoms))])
        position1_new = np.array([molecule1.getAtomByMolIndex(i).position for i in range(len(molecule1.atoms))])
        np.testing.assert_allclose(position1_new[0:3,:],position2_old[0:3,:],err_msg="swap_molecule_positions does not correctly trade the coordinates of the common atoms.")
        
    def test_swap_molecule_positions_maintains_bond_angles(self):
        molecule1,molecule2=self.swap_move.select_random_molecules()
        atoms1 = [molecule1.getAtomByMolIndex(i) for i in range(len(molecule1.atoms))]
        atoms2 = [molecule2.getAtomByMolIndex(i) for i in range(len(molecule2.atoms))]
        angles1_pre = getAngles(atoms1[1:5],2)
        angles2_pre = getAngles(atoms2[1:],10)       
        self.swap_move.swap_molecule_positions(molecule1,molecule2)
        angles1_post = getAngles(atoms1[1:5],2)
        angles2_post = getAngles(atoms2[1:],10) 
        np.testing.assert_allclose(angles1_pre,angles1_post,err_msg='swap_molecule_positions changes bond angles in molecule 1')  
        np.testing.assert_allclose(angles2_pre,angles2_post,err_msg='swap_molecule_positions changes bond angles in molecule 2')

    @mock.patch.object(mvc.CBMCSwap,'regrow')
    def test_move_returns_False_when_regrow_returns_False(self,mock_method):
        mock_method.return_value = False
        self.assertFalse(self.swap_move.move(),msg="CBMCSwap move method does not return False when regrow method returns False")

    @mock.patch.object(mvc.CBMCSwap,'regrow')
    def test_move_returns_False_when_regrow_returns_False(self,mock_method):
        mock_method.return_value = 1
        self.assertTrue(self.swap_move.move(),msg="CBMCSwap move method does not return True when ratio of regrow weights is 1")

    def test_all_regrowths_succesful_for_1ddt_1meoh_on_a_nanoparticle(self):
        molecule1,molecule2 = self.swap_move.select_random_molecules()
        log_Wo_chain1 = self.swap_move.regrow(molecule1,self.swap_move.starting_index,keep_original=True)
        log_Wo_chain2 = self.swap_move.regrow(molecule2,self.swap_move.starting_index,keep_original=True)
        self.swap_move.swap_molecule_positions(molecule1,molecule2)
        log_Wf_chain1 = self.swap_move.regrow(molecule1,self.swap_move.starting_index,keep_original=False)
        log_Wf_chain2 = self.swap_move.regrow(molecule2,self.swap_move.starting_index,keep_original=False)
        self.assertTrue(all([log_Wf_chain1,log_Wf_chain2,log_Wo_chain1,log_Wo_chain2]),msg="On a bare surface not all regrowths return valid Rosenbluth weights.")

    def tearDown(self):
        os.chdir(script_path)

def getAngles(atoms,numAngles):
    angles = np.empty(numAngles)
    for i in range(numAngles):
        atom1 = atoms[0+i].position
        atom2 = atoms[1+i].position
        atom3 = atoms[2+i].position
        line1 = atom1-atom2
        line2 = atom3-atom2
        angles[i] = np.arccos(np.dot(line1, line2) / (np.linalg.norm(line1)*np.linalg.norm(line2)))
    return angles







