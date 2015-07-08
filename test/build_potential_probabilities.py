#!/usr/bin/python
import numpy as np
from math import *
import random as rnd
import matplotlib.pyplot as plt

'''
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
'''

def bondPeClass2(r,r0,k2,k3,k4):
	return (k2*(r-r0)**2+k3*(r-r0)**3+k4*(r-r0)**4)

def anglePeClass2(theta,theta0,k2,k3,k4):
	Ea = k2*(theta-theta0)**2+k3*(theta-theta0)**3+k4*(theta-theta0)**4
	Ebb=0
	Eba=0
	return Ea

def dihedralPeClass2(phi,k1,phi1_0,k2,phi2_0,k3,phi3_0):
	Ed = k1*(1-np.cos(phi-phi1_0))+k2*(1-np.cos(2*phi-phi2_0))+k3*(1-np.cos(3*phi-phi3_0))
	Embt=0
	Eebt=0
	Eat=0
	Eaat=0
	Ebb13=0
	return Ed

if __name__=='__main__':
	#potential_function(2)
	r = np.arange(5,step=0.01)
	theta = np.arange(pi,step=0.001)
	phi = np.arange(2*pi,step=0.001)
	cc_bond_energy = bondPeClass2(r,1.53,299.67,-501.77,679.81)
	sc_bond_energy = bondPeClass2(r,1.823,225.277,-327.706,488.972)
	oc_bond_energy = bondPeClass2(r,1.42,400.395,-835.195,1313.01)
	oh_bond_energy = bondPeClass2(r,0.965,532.506,-1282.9,2004.77)

	ccc_angle_energy = anglePeClass2(theta,112.67*pi/180,39.516,-7.443,-9.5583)
	scc_angle_energy = anglePeClass2(theta,112.564*pi/180,47.0276,-10.679,-10.1687)
	coh_angle_energy = anglePeClass2(theta,105.8*pi/180,52.7061,-12.109,-9.8681)
	cco_angle_energy = anglePeClass2(theta,111.27*pi/180,54.5381,-8.3642,-13.0838)

	cccc_dihedral_energy = dihedralPeClass2(phi,0,0,0.514,0,-0.143,0)
	sccc_dihedral_energy = dihedralPeClass2(phi,-0.7017,0,0.0201,0,0.104,0)
	scco_dihedral_energy = dihedralPeClass2(phi,0 ,0 ,0, 0, 0.158, 0)
	ccoh_dihedral_energy = dihedralPeClass2(phi,-0.6732, 0, -0.4778, 0, -0.167, 0)
	
	pot_plot = np.hstack((phi.reshape((phi.shape[0],1)),cccc_dihedral_energy.reshape((phi.shape[0],1))))
	print pot_plot
	np.savetxt('dih_potential_1',pot_plot,header = "distance(A)\tpotential(kcal/mol)")	
	'''
	pdf = np.nan_to_num(np.exp(-cccc_dihedral_energy))
	print pdf
	cdf_unnorm = np.cumsum(pdf)
	#print cdf_unnorm
	cdf = cdf_unnorm/cdf_unnorm[-1]
	#prob = rnd.uniform(0,1)
	#chosen_index = np.searchsorted(cdf,prob)
	#chosen_r = r[chosen_index]
	tries=100000
	chosen_rs = np.empty([tries])
	for i in xrange(tries):
		prob = rnd.uniform(0,1)
		chosen_index = np.searchsorted(cdf,prob)
		chosen_rs[i] = phi[chosen_index]
		if((i+1)%1000==0):
			print "On random try "+str(i)
	#(hist,bin_edges) = np.histogram(chosen_rs,bins=int(tries/10),normed=True)
	#bins = (bin_edges[1:]+bin_edges[:-1])/2
	fig, ax1 = plt.subplots()
	ax1.hist(chosen_rs,bins=int(tries/10))
	ax2 = ax1.twinx()
	ax2.plot(phi,pdf)
	plt.show()
	'''
