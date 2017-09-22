#!/usr/bin/python
import sys
import numpy as np
import string
from math import *
import random as rnd
import time
from subprocess import call

#import pandas as pd

def readfile(filename):
	start = time.time()
	input = open(filename,"r",8192)
	print "reading file " + filename
	currentId=0
	startPos=1
	end = time.time()
	timeOpening = end-start
	print "Readfile took "+str(timeOpening)+"secs to open file"
	start = time.time()
	line = input.readline()
	out = line.find("atoms")
	while(out==-1):
		line = input.readline()
		out = line.find("atoms")
		startPos+=1
	print line
	end = time.time()
	timeSeeking = end-start
	print "Readfile took "+str(timeSeeking)+"secs to find #atoms"
	numwords = line.split()
	numatms = int(numwords[0])
	atoms = np.zeros((numatms,7))
	start = time.time()
	while(input.readline().find("Atoms")==-1):
		startPos+=1
		continue
	input.readline()
	startPos+=1
	end = time.time()
	timeSeeking = end-start
	print "Readfile took "+str(timeSeeking)+"secs to find Atoms"
	start = time.time()
	#atoms = pd.read_csv(filename,engine='c',skiprows=startPos,nrows=numatms,skipinitialspace=True,delim_whitespace=True)
	#print atoms
	#line=input.readline()
	#lines=input.readlines(numatms)
	for j in range(numatms):
		line=input.readline()
	#	print "Processing line "+str(j)
	#	print line
		record = line.split()
		atoms[j,0]=int(record[0])
		atoms[j,1]=int(record[1])
		atoms[j,2]=int(record[2])
		atoms[j,3]=float(record[3])
		atoms[j,4]=float(record[4])
		atoms[j,5]=float(record[5])
		atoms[j,6]=float(record[6])
		#for i in range(7):
		#	if i<3:
		#		atoms[j,i]=int(record[i])
		#	else:
		#		atoms[j,i]=float(record[i])
	input.close()
	end = time.time()
	timeReading = end-start
	print "Readfile took "+str(timeReading)+"secs to read Atoms"
	return atoms

def shiftMolecule(atoms,indices,shiftx,shifty,shiftz):
        numatms=len(indices)
	#print "Starting shift by " + str(shiftx)+"x, "+str(shifty)+"y and "+str(shiftz)+"original positions are:"
	#print(atoms[indices,:])
        for i in range(numatms):
                atoms[indices[i],4]+=shiftx
                atoms[indices[i],5]+=shifty
                atoms[indices[i],6]+=shiftz
	#print "finished shifting atoms the new coordinates are:"
	#print(atoms[indices,:])
def rotateMolecule(atoms,indices,theta,phi):
        numatoms = len(indices)
        theta = theta
        phi = phi
        rotateZ = np.array([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])
        rotateY = np.array([[cos(phi),0,sin(phi)],[0,1,0],[-sin(phi),0,cos(phi)]])
        for i in range(numatoms):
                firstRotate = np.dot(rotateY,np.transpose(atoms[indices[i],[4,5,6]]))
                secondRotate = np.dot(rotateZ,np.transpose(atoms[indices[i],[4,5,6]]))
                atoms[indices[i],[4,5,6]]=np.transpose(secondRotate)

def calcCOM(atoms,indices):
	mass = [107.8682,14,15,32.065,16,1]
	numatms = len(indices)
	totalmass=0
	x=0
	y=0
	z=0
	for i in range(numatms):
		atommass = mass[int(atoms[indices[i],2])-1]
		#print "Atom is of type "+str(int(atoms[indices[i],2]))+" and a mass of "+str(atommass)+"with coordinates "+str(atoms[indices[i],4])+"x, "+str(atoms[indices[i],5])+"y and "+str(atoms[indices[i],6])+"z"
		x+=atommass*atoms[indices[i],4]
		y+=atommass*atoms[indices[i],5]
		z+=atommass*atoms[indices[i],6]
		totalmass+=atommass
	#print "COM is "+str(x/totalmass)+"x, "+str(y/totalmass)+"y and "+str(z/totalmass)
	return np.array([x/totalmass,y/totalmass,z/totalmass])	

def rotateAxis(olddir,newdir):
        axis = np.cross(olddir,newdir)
        axis = axis/np.linalg.norm(axis)
        theta = acos(np.dot(olddir,newdir)/(np.linalg.norm(olddir)*np.linalg.norm(newdir)))
        skewmat = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.identity(3) + sin(theta)*skewmat+(1-cos(theta))*np.linalg.matrix_power(skewmat,2)
        return R

def comRotation(a,b):
	v = np.cross(a,b)
	vX = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	s = np.linalg.norm(v)
	c = np.dot(a,b)
	print "normalization is "+str(((1-c)/(s**2)))
	R = np.identity(3) + vX +np.linalg.matrix_power(vX,2)*((1-c)/(s**2))
	return R

def randomRotate(atoms,molId):
	max_angle = 0.174532925
	sulfur = atoms[(atoms[:,2]==4)&(atoms[:,1]==molId)]
	indices = np.where((atoms[:,1]==molId))[0]
	r = sqrt(sulfur[0,4]**2+sulfur[0,5]**2+sulfur[0,6]**2)
	theta = acos(sulfur[0,6]/r)
	phi = atan2(sulfur[0,5],sulfur[0,4])
	rAxis = np.array([sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)])
	thetaAxis = np.array([cos(theta)*cos(phi),cos(theta)*sin(phi),-sin(theta)])
	phiAxis = np.array([-sin(phi),cos(phi),0])
	rotAxis = rnd.choice((rAxis,thetaAxis,phiAxis))
	if(np.array_equal(rotAxis,rAxis)):
		angle = rnd.uniform(-max_angle,max_angle)
		print "Rotating about radial axis"
	else:
		angle = rnd.uniform(-max_angle,max_angle)
		print "Rotating about theta or phi axis"
	rotMatrix = np.array([cos(angle)+(rotAxis[0]**2)*(1-cos(angle)),rotAxis[0]*rotAxis[1]*(1-cos(angle))-rotAxis[2]*sin(angle),rotAxis[0]*rotAxis[2]*(1-cos(angle))+rotAxis[1]*sin(angle),rotAxis[0]*rotAxis[1]*(1-cos(angle))+rotAxis[2]*sin(angle),cos(angle)+(rotAxis[1]**2)*(1-cos(angle)),rotAxis[1]*rotAxis[2]*(1-cos(angle))-rotAxis[0]*sin(angle),rotAxis[0]*rotAxis[2]*(1-cos(angle))-rotAxis[1]*sin(angle),rotAxis[1]*rotAxis[2]*(1-cos(angle))+rotAxis[0]*sin(angle),cos(angle)+(rotAxis[2]**2)*(1-cos(angle))])
	rotMatrix = np.reshape(rotMatrix,(3,3))
	#print "Rotating "+str(angle)+" radians"
	atoms[indices,4:7]-=sulfur[0,4:7]
	#print "Atoms should now be originated"
	#print atoms[indices,4:7]
	for i in xrange(indices.shape[0]):
		atoms[indices[i],4:7]=np.dot(rotMatrix,np.transpose(atoms[indices[i],4:7]))
	atoms[indices,4:7]+=sulfur[0,4:7]
	
def swapMolecules(molId1,molId2,atoms,centerRotation):
	atoms1 = np.where((atoms[:,1]==molId1))[0]
	atoms2 = np.where((atoms[:,1]==molId2))[0]
	sulfur1 = atoms[(atoms[:,1]==molId1) & (atoms[:,2]==float(4))]
	sulfur2 = atoms[(atoms[:,1]==molId2) & (atoms[:,2]==float(4))]
	x1 = sulfur1[0,4]-centerRotation[0]
	x2 = sulfur2[0,4]-centerRotation[0]
	y1 = sulfur1[0,5]-centerRotation[1]
        y2 = sulfur2[0,5]-centerRotation[1]
	z1 = sulfur1[0,6]-centerRotation[2]
        z2 = sulfur2[0,6]-centerRotation[2]
	phi1 = atan2(y1,x1)
	phi2 = atan2(y2,x2)
	theta1 = acos(z1/sqrt(x1**2+y1**2+z1**2))
	theta2 = acos(z2/sqrt(x2**2+y2**2+z2**2))
	shiftX = x2 - x1
	shiftY = y2 - y1
	shiftZ = z2 - z1
	shiftTheta = theta2-theta1
	shiftPhi = phi2-phi1
	shiftMolecule(atoms,atoms1,shiftX,shiftY,shiftZ)
	shiftMolecule(atoms,atoms2,-shiftX,-shiftY,-shiftZ)
	com1=calcCOM(atoms,atoms1)
	a1 = com1-sulfur2[0,4:7]
	b1 = np.array([sin(theta2)*cos(phi2),sin(theta2)*sin(phi2),cos(theta2)])
	com2=calcCOM(atoms,atoms2)
	a2 = com2-sulfur1[0,4:7]
	b2 = np.array([sin(theta1)*cos(phi1),sin(theta1)*sin(phi1),cos(theta1)])
	atoms[atoms1,4:7]-=sulfur2[0,4:7]
	for i in range(len(atoms1)):
		atoms[atoms1[i],4:7]=np.dot(rotateAxis(a1,b1),np.transpose(atoms[atoms1[i],[4,5,6]]))
	atoms[atoms1,4:7]+=sulfur2[0,4:7]
	atoms[atoms2,4:7]-=sulfur1[0,4:7]
	for i in range(len(atoms2)):
        	atoms[atoms2[i],4:7]=np.dot(rotateAxis(a2,b2),np.transpose(atoms[atoms2[i],[4,5,6]]))
        atoms[atoms2,4:7]+=sulfur1[0,4:7]

def editFile(filein,atoms,fileout):
	input = open(filein,"r")
	output = open(fileout,"w")
	currentId=0
	line = input.readline()
	out = line.find("atoms")
	while(out==-1):
		output.write(line)
		line = input.readline()
		out = line.find("atoms")
	output.write(line)
	while(line.find("Atoms")==-1):
		line = input.readline()
		output.write(line)
	output.write("\n")
	np.savetxt("atoms.temp",atoms,"%7d %7d %4d %10.6f %16.8f %16.8f %16.8f")
	tempfile = open("atoms.temp","r")
	output.write(tempfile.read())
	while(input.readline().find("Bonds")==-1):
		continue
	output.write("\nBonds\n")
	while(line != ''):
		line = input.readline()
		output.write(line)
	input.close()
	output.close()
	tempfile.close()	

def molCollide(atoms,indices):
	numatms = len(indices)
	xcenter = 40.9
	ycenter = 40.9
	zcenter = 40.9
	for i in range(numatms):
		x = atoms[indices[i],4]
		y = atoms[indices[i],5]
		z = atoms[indices[i],6]
		#print "x is "+str(x)+" y is "+str(y)+" z is "+str(z)
		if sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)<20.45:
			return True
	return False

def getPotential(filename):
	pefile = open(filename,"r")
	lines=pefile.readlines()
	value=0.0
	try:
		return float(lines[len(lines)-1].split()[1])
	except ValueError:
		return 0.0

if __name__ == "__main__":
	inputfile=sys.argv[1]
	pefile = open("Potential_Energy.txt","w")
	basename=inputfile.split('.')[0]
	atoms = readfile(inputfile)
	kb=0.0019872041 #in kcal/mol-K
	T=5000
	beta = 1.0/(kb*T)
	molIds = atoms[atoms[:,2]==4][:,1]
	centerRotation=np.array([40.9,40.9,40.9])
	numsteps=40000
	dT = (T-200.0)/numsteps
	print "Center of rotation is "+str(centerRotation[0])+"x, "+str(centerRotation[1])+"y and "+str(centerRotation[2])+"z"
	rnd.seed()
	collisions=0
	lastenergy=getPotential("pe.out")
	pefile.write("Step\tEnergy(kcal/mol)\n")
	timeOpening = 0.0
	timeSwapping = 0.0
	timeRotating = 0.0
	timeWriting = 0.0
	timeChoosing = 0.0
	sim_time=0.0
	swapMoves = 0.0001
	rotateMoves = 0.0001
	for i in xrange(numsteps):
		#timeOpening = 0.0
		#timeSwapping = 0.0
		#timeRotating = 0.0
		#timeWriting = 0.0
		#timeChoosing = 0.0
		#swapMoves = 0
		#rotateMoves = 0
		start = time.clock()
		pefile.write(str(i)+"\t"+str(lastenergy)+"\n")
		pefile.flush()
		#editFile(inputfile,atoms,"preswap.lmp")
		old_atoms = np.copy(atoms)	
		move = rnd.choice(("swap","rotate"))
		end = time.clock()
		timeOpening+=end-start
		if move is "swap":
			swapMoves+=1
			start = time.clock()
			swappedList = rnd.sample(molIds,2)
			atoms1 = atoms[(atoms[:,1]==swappedList[0])]
			atoms2 = atoms[(atoms[:,1]==swappedList[1])]
			while(((3 in atoms1[:,2]) & (3 in atoms2[:,2])) | ((5 in atoms1[:,2]) & (5 in atoms2[:,2]))):
				swappedList = rnd.sample(molIds,2)
				atoms1 = atoms[(atoms[:,1]==swappedList[0])]
				atoms2 = atoms[(atoms[:,1]==swappedList[1])]
			swapMolecules(swappedList[0],swappedList[1],atoms,centerRotation)
			if molCollide(atoms,np.where((atoms[:,1]==swappedList[0]))[0]) or  molCollide(atoms,np.where((atoms[:,1]==swappedList[1]))[0]):
				print "molecule collided"
				#continue
			end = time.clock()
			timeSwapping+= end-start
		else:
			rotateMoves+=1
			start = time.clock()
			molId = rnd.sample(molIds,1)
			randomRotate(atoms,molId)
			end = time.clock()
			timeRotating+= end-start
		start = time.clock()
		editFile(inputfile,atoms,basename+str(i)+".lmp")
		call(["cp",basename+str(i)+".lmp",basename+".lmp"])
		sim_start = time.clock()	
		call("./runlammps.sh")
		sim_end = time.clock()
		sim_time+= sim_end-sim_start
		call(["cp","ddt_me_200.xyz","ddt_me_step"+str(i)+".xyz"])
		energy=getPotential("pe.out")
		end = time.clock()
		timeWriting+=end-start
		dU = energy-lastenergy
		start = time.clock()
		if(dU<=0):
			#atoms=readfile(basename+str(i)+".lmp")
			lastenergy=energy
			print "Move accepted energy diff is: "+str(dU)
		elif(rnd.random()<exp(-beta*dU)):
			#atoms=readfile(basename+str(i)+".lmp")
			lastenergy=energy
			print "Move accepted energy diff is: "+str(dU)
		else:
			#atoms=readfile("preswap.lmp")
			atoms = old_atoms
			print "Move rejected energy diff is: "+str(dU)
		T = (T-dT)
		beta = 1.0/(kb*T)
		end = time.clock()
		timeChoosing += end-start
		#print "This loop spent "+str(timeOpening)+"secs opening, "+str(timeSwapping)+"secs swapping, "+str(timeRotating)+"secs rotating, "+str(timeWriting)+"secs writing, "+str(timeChoosing)+"secs choosing"
		print "The average time spent is "+str(timeOpening/float(i+1))+"secs opening, "+str(timeSwapping/float(swapMoves))+"secs swapping, "+str(timeRotating/float(rotateMoves))+"secs rotating, "+str(timeWriting/float(i+1))+"secs writing, "+str(sim_time/float(i+1))+"secs in sim, "+str(timeChoosing/float(i+1))+"secs choosing"
	print "Xmin = "+str(np.amin(atoms[:,4]))
        print "Xmax = "+str(np.amax(atoms[:,4]))
        print "Xmin = "+str(np.amin(atoms[:,5]))
        print "Ymax = "+str(np.amax(atoms[:,5]))
        print "Zmin = "+str(np.amin(atoms[:,6]))
        print "Zmax = "+str(np.amax(atoms[:,6]))
