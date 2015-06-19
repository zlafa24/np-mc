#!/usr/bin/python
import numpy as np
from math import *

def potential_function(maxrange):
	r0 = 1.53
	k2 = 299.67
	k3 = -501.77
	k4 = 679.81
	step = 0.01
	
	start = r0-(maxrange-r0)
	energies = np.empty([int((maxrange-start)/step),2]) 
	for i in xrange(int((maxrange-start)/step)):
		r = step*i+start
		energies[i,0] = r
		energies[i,1] = np.exp(-(k2*(r-r0)**2+k3*(r-r0)**3+k4*(r-r0)**4))
	energies[:,1] = np.cumsum(energies[:,1])
	energies[:,1] = energies[:,1]/energies[-1,1]
	print energies[:,1]
	np.savetxt('pot_prob.txt',energies,header="distance\tpotential")

if __name__=='__main__':
	potential_function(2)

