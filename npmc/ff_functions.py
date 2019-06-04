from math import *
def opls(phi,parameters):
    (k1,k2,k3,k4)=parameters
    return 0.5*k1*(1+cos(phi))+0.5*k2*(1-cos(2*phi))+0.5*k3*(1+cos(3*phi))+0.5*k4*(1-cos(4*phi))

def harmonic(theta,theta0,k):
    return k*(theta-theta0)**2