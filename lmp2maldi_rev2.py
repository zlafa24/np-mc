#!/home/snm8xf/anaconda/bin/python

import numpy as np
from read_lmp_rev6 import *
import sys
import random as rnd
from math import *

if __name__ == "__main__":
	numsamples = 100000
	bin0 = 0
	bin1 = 0
	bin2 = 0
	bin3 = 0
	bin4 = 0
	bin5 = 0
	filename = sys.argv[1]
	molecules = readAll(filename)
	sulfurs = molecules[0][(molecules[0][:,2].astype(int)==4)]
        molIds = np.unique(sulfurs[:,1])
	ddtIds = molecules[0][(molecules[0][:,2].astype(int)==3)][:,1]
	meohIds = molecules[0][(molecules[0][:,2].astype(int)==5)][:,1]
	for i in xrange(numsamples):	
		if((i+1)%5000==0):
			print "Processed "+str(i+1)+" samples so far and spectra looks like this: \n"
			print "Fragment0 Fragment1 Fragment2 Fragment3 Fragment4"
			total = float(bin0+bin1+bin2+bin3+bin4+bin5)
			frac0 = bin0/total
			frac1 = bin1/total
			frac2 = bin2/total
			frac3 = bin3/total
			frac4 = bin4/total
			frac5 = bin5/total
			print str(frac0)+" "+str(frac1)+" "+str(frac2)+" "+str(frac3)+" "+str(frac4)+" "+str(frac5)+"\n"+"\n"
		rndId = rnd.sample(molIds,1)

		rndMol = sulfurs[(sulfurs[:,1]==rndId)]
		curX = rndMol[0,4]
		curY = rndMol[0,5]
		curZ = rndMol[0,6]

		#print "Selected sulfur is at "+str(curX)+"x,"+str(curY)+"y, and "+str(curZ)+"z"

		#print "This gives a radial distance of "+str(sqrt((curX-40.9)**2+(curY-40.9)**2+(curZ-40.9)**2))+" Angstroms"
		sulfurs[:,3]=(sulfurs[:,4]-curX)**2+(sulfurs[:,5]-curY)**2+(sulfurs[:,6]-curZ)**2
		sulfurs[:,3]=np.sqrt(sulfurs[:,3])
		#print sulfurs[:,3]
		sorted_sulfurs = sulfurs[sulfurs[:,3].argsort()]
		rndNN = rnd.sample(np.array([0,1,2,3,4,5]),4)
		#print sulfurs[rndNN,3]
		chosenIds = np.array([rndId,sorted_sulfurs[0,1],sorted_sulfurs[1,1],sorted_sulfurs[2,1],sorted_sulfurs[3,1]])
		#chosenIds = np.array([rndId,sulfurs[rndNN[0],1],sulfurs[rndNN[1],1],sulfurs[rndNN[2],1],sulfurs[rndNN[3],1]])
		chosenOnes = molecules[0][(molecules[0][:,1].astype(int)==chosenIds[0])|(molecules[0][:,1].astype(int)==chosenIds[1])|(molecules[0][:,1].astype(int)==chosenIds[2])|(molecules[0][:,1].astype(int)==chosenIds[3])|(molecules[0][:,1].astype(int)==chosenIds[4])]

		numMeoh=0
		numDdt=0
		for i in range(5):
			curMol=chosenOnes[(chosenOnes[:,1].astype(int)==chosenIds[i])][0,1]
			#print "Analyzing following molecule:"
			#print curMol
			if((curMol in meohIds)):
				#print chosenOnes[(chosenOnes[:,1].astype(int)==chosenIds[i])]
				numMeoh+=1
			else:
				numDdt+=1
		#print "There are "+str(numMeoh)+" mercaptoethanols and "+str(numDdt)+" dodecanethiols"
		if(numDdt==0):
			bin0+=1
		elif(numDdt==1):
                        bin1+=1
		elif(numDdt==2):
                        bin2+=1
		elif(numDdt==3):
                        bin3+=1
		elif(numDdt==4):
                        bin4+=1
		elif(numDdt==5):
                        bin5+=1
total = float(bin0+bin1+bin2+bin3+bin4+bin5)
frac0 = bin0/total
frac1 = bin1/total
frac2 = bin2/total
frac3 = bin3/total
frac4 = bin4/total
frac5 = bin5/total
print frac0
print frac1
print frac2
print frac3
print frac4
print frac5

