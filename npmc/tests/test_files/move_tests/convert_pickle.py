import numpy as np
import pickle
import sys

filename = sys.argv[1]

result = pickle.load(open(filename,'rb'))
pickle.dump(result,open(filename,'wb'),protocol=2)
