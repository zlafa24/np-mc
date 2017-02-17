#!/usr/bin/python

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from math import *

#Generic 1/sqrt(x) function for scipy to fit to data
def invsqrt(x,a,b):
	return a/np.sqrt(x)-b

#Loads energy file assumes data is in column 1 which is the second column
energies = np.loadtxt("Potential_Energy.txt",skiprows=1,usecols=(1,))

#Bins the data into 400 bins of 1000 data points each, change this to match the binning you want
energies=energies.reshape(400,1000)
stds=np.std(energies,axis=1)

print "Shape of stds "+str(stds.shape)

#Fits function and plots the real data and the fit against each other
params=opt.curve_fit(invsqrt,np.arange(len(stds))[1:],stds[1:])
plt.plot(np.arange(len(stds))[1:],stds[1:],np.arange(len(stds))[1:],invsqrt(np.arange(len(stds))[1:],params[0][0],params[0][1]))
plt.show()
