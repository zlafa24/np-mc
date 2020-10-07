from math import *
import numpy as np

def opls(phi,parameters):
    (k1,k2,k3,k4)=parameters
    return 0.5*k1*(1+np.cos(phi))+0.5*k2*(1-np.cos(2*phi))+0.5*k3*(1+np.cos(3*phi))+0.5*k4*(1-np.cos(4*phi))

def fourier(phi,parameters):
    pa = np.zeros(4*3)
    n_total = parameters[0]
    first_n = parameters[2]
    for i in np.arange(first_n,4+first_n):
        pa[int(i*3):int(i*3+3)] = parameters[int(i*3+1):int(i*3+4)]
    return pa[0]*(1+np.cos(pa[1]*phi-pa[2]))+pa[3]*(1+np.cos(pa[4]*phi-pa[5]))+pa[6]*(1+np.cos(pa[7]*phi-pa[8]))+pa[9]*(1+np.cos(pa[10]*phi-pa[11]))

def harmonic(theta,parameters):
    (k,theta0) = parameters
    theta0 = np.radians(theta0)
    return k*(theta-theta0)**2

def cvff(phi,parameters):
    (K,d,n) = parameters
    return K*(1+d*np.cos(n*phi))