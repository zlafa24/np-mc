import os,sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.abspath(script_path+'/../src'))

import maldi_class as mldc
from subprocess import check_output

data_file = sys.argv[1]

maldi = mldc.MALDISpectrum(data_file,anchor_type=4,numsamples=50000,nn_distance=8.0,ligands_per_fragment=5,type_lengths=(13,5))
spectrum = maldi.get_maldi_spectrum()
print(spectrum)

