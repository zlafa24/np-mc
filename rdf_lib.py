#!/usr/bin/python
import fileinput as fin
import re
from subprocess import call
import sys
import read_lmp_rev6 as rdlmp
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate

def get_rdf(filename):  
    oxygen_id = 5
    max_mol_id = 2553
    molecules = rdlmp.readAll(filename)
    atoms = molecules[0]
    oxygens = atoms[(atoms[:,2]==oxygen_id) & (atoms[:,1]<=max_mol_id)]
    num_oxygens = oxygens.shape[0]

    dists = np.zeros(num_oxygens**2)

    area = 4*pi*40.0**2

    rho_norm = num_oxygens/area
    
    for i,oxygen in enumerate(oxygens):
        for j,oxygen2 in enumerate(oxygens):
            if (i*num_oxygens+j+1)%10000==0:
                print "Processed "+str(i*num_oxygens+j)+" pair out of "+str(num_oxygens**2)
            dists[i*num_oxygens+j] = np.linalg.norm(oxygen[4:7]-oxygen2[4:7])
    
    numbins=50

    [counts,bins] = np.histogram(dists,bins=numbins)
    dr = bins[1]-bins[0]

    normcounts = np.zeros(counts.shape[0])

    for i,(r,count) in enumerate(zip(bins,counts)):
        normcounts[i] = count/(rho_norm*(2*pi*r)*dr) if r!=0 else 0

    return (normcounts,bins)

def compare_rdf(infile1,infile2):
    (counts1,bins1) = get_rdf(infile1)
    (counts2,bins2) = get_rdf(infile2)
    trans1 = np.fft.fft(counts1)
    trans2 = np.fft.fft(counts2)
    xs = np.linspace(0,50,50)
    dx1 = (bins1[1]-bins1[0])
    dx2 = (bins2[1]-bins2[0])
    x1 = xs
    x2 = xs*dx1/dx2

    real1 = np.real(np.array([sum(trans1*np.exp(2*pi*1j*np.fft.fftfreq(50)*x)) for x in x1[0:20]]))/21.0
    real2 = np.real(np.array([sum(trans2*np.exp(2*pi*1j*np.fft.fftfreq(50)*x)) for x in x2[0:20]]))/21.0

    r2 = sum(((real1-real2)/max(real1))**2)*dx1
    return r2  

def spline_rdf(rdf,xs):
    (normcounts,bins)=rdf
    tck = interpolate.splrep(bins[0:-1],normcounts)
    newcounts = interpolate.splev(xs,tck)
    return (xs,newcounts)
