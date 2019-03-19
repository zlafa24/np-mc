#!/usr/bin/python3
import pickle
import numpy as np
import sys
import npmc.forcefield_class as forcefield_class

filename = sys.argv[1]

result = pickle.load(open(filename,'rb'),encoding='latin1')
pickle.dump(result,open(filename,'wb'))
