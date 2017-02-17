#!/home/snm8xf/anaconda/bin/python
from rdf_lib import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __name__=="__main__":
    xs = np.linspace(0,45,200)
    start = int(sys.argv[1])
    final = int(sys.argv[2])
    samples = final-start
    rdf_accum = np.zeros(xs.size)
    path = os.getcwd()
    print "Working from path"+path
    for i in range(start,final+1):
        if((i+1)%10==0):
            print "On step "+str(i)
        smooth_rdf =  spline_rdf(get_rdf(path+"/configuration_"+str(i)+"99.lmp"),xs)
        rdf_accum+=smooth_rdf[1]
    rdf_ave=rdf_accum/float(samples)
    np.savetxt("avrdf.txt",np.column_stack((xs,rdf_ave)))
    plt.plot(xs,rdf_ave)
    plt.show()    
