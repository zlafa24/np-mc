    def test_evalute_trial_rotations_selects_correct_dihedral_angle_distribution(self):
        molecule = self.cbmc_move.simulation.molecules[1]
        index = 3
        self.cbmc_move.turn_off_molecule_atoms(self.cbmc_move.simulation.molecules[1],index)
        num_attempts = 500000
        (thetas,dtheta) = np.linspace(0,2*pi,num=500,retstep=True)
        expected_pdf = np.multiply(self.dihedral_type3_pdf[1],self.SCCO_pair_pdf)/(sum(np.multiply(self.dihedral_type3_pdf[1],self.SCCO_pair_pdf))*dtheta)
        import pdb;pdb.set_trace()
        rotations = np.empty(num_attempts)
        for i in range(num_attempts):
            rotations[i] = self.cbmc_move.evaluate_trial_rotations(molecule,index)
        (normed_histogram,bins) = np.histogram(rotations,bins=50,density=True) 
        import pdb;pdb.set_trace()
        np.testing.assert_array_almost_equal(normed_histogram,expected_pdf)
        #np.savetxt(open(os.path.abspath('./actual_SCCO_dihedral_PDF.txt'),'rw'),rotations)
 
