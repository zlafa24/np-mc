#!/usr/bin/python
from subprocess import call
import numpy as np
import sys

numsteps = 399
dist=sys.argv[1]
outfile = open(sys.argv[2],'w')
outfile.write('#clusters\taverage\tmedian\tmax')
outfile.close()
outfile = open(sys.argv[2],'a')

for i in range(numsteps):
	if((i+1)%10==0):
		print "On step "+str(i+1)
	call(["clusters_rev2.py","configuration_"+str(i+1)+"99.lmp",dist],stdout=outfile)
