from math import *
import numpy as np

def opls(phi,parameters):
    (k1,k2,k3,k4)=parameters
    return 0.5*k1*(1+np.cos(phi))+0.5*k2*(1-np.cos(2*phi))+0.5*k3*(1+np.cos(3*phi))+0.5*k4*(1-np.cos(4*phi))

def harmonic(theta,parameters):
    (k,theta0) = parameters
    theta0 = np.radians(theta0)
    return k*(theta-theta0)**2