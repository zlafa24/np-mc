#!/usr/bin/python
import numpy as np
from math import *
from read_lmp_rev6 import *
import sys, getopt

try:
	opts,args = getopt.getopt(sys.argv[1:],"n:f:c:s:",["coordFile="])
except getopt.GetoptError:
      	print 'Error in provided options'
      	sys.exit(2)
print opts
print args

r = 25
dPhi=-1 #original 0.245

for opt,arg in opts:
	print "Option "+str(opt)+" passed with arg "+str(arg)
	if(opt == '-n'):
		numLigands = int(arg)
	elif(opt=='-f'):
		coordFile = arg
	elif(opt=='-c'):
		config = arg
	elif(opt=='-s'):
		dPhi = float(arg)
if(dPhi==-1):
    dPhi = sqrt((4*pi*r**2)/(numLigands*r**2))
    print "Necessary dPhi is calculated to be "+str(dPhi)
xyzfile = "addmolecule.xyz"
lammp_template = "addmolecule_24_rand.lmp"
center = [61.35,61.35,61.35]
molId=200
ocharge=-0.685
hcharge=0.40
molecules=readAll(lammp_template)
nummol=int(np.amax(molecules[0][:,1]))
atoms=molecules[0]
bonds=molecules[1]
angles=molecules[2]
diheds=molecules[3]
molatoms=atoms[atoms[:,1]==molId][:,0]
silveratoms=atoms[atoms[:,2]==1]
blankmolecule=(np.empty([0,7]),np.empty([0,4]),np.empty([0,5]),np.empty([0,6]))
silvercoords = np.loadtxt(coordFile,skiprows=2,usecols=(1,2,3))
config_list = np.zeros([numLigands])
if(config=='random'):
	config_list[0:(numLigands/2)]=1
	np.random.shuffle(config_list)
elif(config=='stripe'):
	config_list[:]=1
	config_list[(np.arange(len(config_list))%30)<15]=0
	print "Config list is "+str(config_list)
elif(config=='janus'):
	config_list[0:(numLigands/2)]=1
silveratoms=atoms[(atoms[:,2]==1)]
print "x of silver atoms is "+str(np.amin(silveratoms[:,4]))+"-"+str(np.amax(silveratoms[:,4]))
print "y of silver atoms is "+str(np.amin(silveratoms[:,5]))+"-"+str(np.amax(silveratoms[:,5]))
print "z of silver atoms is "+str(np.amin(silveratoms[:,6]))+"-"+str(np.amax(silveratoms[:,6]))

def checkCollision(newcoords,atoms):
		dists = [np.linalg.norm(newcoords-coord) for coord in atoms[:,4:7] ]
		collisions=[dist for dist in dists if dist<2.8]
		if(len(collisions)>0):
			print "Collision imminent"
			print "collsions are:"+str(collisions)
		return (len(collisions)>0)

def selectMolecule(molecules,molId): 
	molBonds=np.array([])
	molAngles=np.array([])
	molDiheds=np.array([])
	atoms=molecules[0]
	bonds=molecules[1]
	angles=molecules[2]
	diheds=molecules[3]
	molatoms=atoms[atoms[:,1]==molId][:,0]
	for atom in molatoms:
		molBonds=np.union1d(molBonds,np.where(((bonds[:,2]==atom)|(bonds[:,3]==atom)))[0])
		molAngles=np.union1d(molAngles,np.where(((angles[:,2]==atom)|(angles[:,3]==atom)|(angles[:,4]==atom)))[0])
		molDiheds=np.union1d(molDiheds,np.where(((diheds[:,2]==atom)|(diheds[:,3]==atom)|(diheds[:,4]==atom)|(diheds[:,5]==atom)))[0])
	molBonds=molBonds.astype(int)
	molAngles=molAngles.astype(int)
	molDiheds=molDiheds.astype(int)

	newatoms=atoms[atoms[:,1]==molId]
	newbonds=np.take(bonds,molBonds,0)
	newangles=np.take(angles,molAngles,0)
	newdiheds=np.take(diheds,molDiheds,0)
	return (newatoms,newbonds,newangles,newdiheds)

phi=0.01
theta=0

print "Number of DDT ligands "+str(molecules[0][(molecules[0][:,2]==3)].shape[0])
print "Number of MEtOH ligands "+str(molecules[0][(molecules[0][:,2]==5)].shape[0])
ddtmols = np.unique(molecules[0][(molecules[0][:,2].astype(int)==3)][:,1])
print ddtmols
meohmols = np.unique(molecules[0][(molecules[0][:,2].astype(int)==5)][:,1])
print meohmols
agmols = np.unique(molecules[0][(molecules[0][:,2].astype(int)==1)][:,1])
ddtinc=0
meohinc=0
molnumber=0
coord_iter=0

for coord in silvercoords:
	newmolecule = selectMolecule(molecules,agmols[0])
	newmolecule[0][:,4:7]=coord
	blankmolecule=addMolecule(blankmolecule,newmolecule)
	coord_iter+=1

row=0
for mol in range(numLigands):
	if(config_list[molnumber]==1):
		newmolecule = selectMolecule(molecules,ddtmols[0])
	else:
		newmolecule = selectMolecule(molecules,meohmols[0])
	molnumber+=1

	atoms = newmolecule[0]
	sulfur = atoms[(atoms[:,2]==4)]
	newtheta =theta+dPhi/sin(phi)
	newcoords = [r*sin(phi)*cos(newtheta)+center[0],r*sin(phi)*sin(newtheta)+center[1],r*cos(phi)+center[2]]
	if(sulfur.size==0):
		continue
	#if(((r*sin(phi)*((2*pi+row*sqrt(3)*cos(30)-newtheta)))<2.8) or checkCollision(newcoords,blankmolecule[0])):
	if(checkCollision(newcoords,blankmolecule[0])):
		row+=1
		theta=row*sqrt(3)*cos(30)
		phi+=dPhi 
	else:
		theta+=dPhi/sin(phi)
	shiftx=r*sin(phi)*cos(theta)+center[0]-sulfur[0,4]
	shifty=r*sin(phi)*sin(theta)+center[1]-sulfur[0,5]
	shiftz=r*cos(phi)+center[2]-sulfur[0,6]
	shiftMolecule(newmolecule[0],range(len(newmolecule[0])),shiftx,shifty,shiftz)
	sulfur = newmolecule[0][(newmolecule[0][:,2]==4)]
	com = calcCOM(newmolecule[0],range(len(newmolecule[0])))
	oldir = com-sulfur[0,4:7]
	newdir = np.array([sin(phi)*cos(theta),sin(phi)*sin(theta),cos(phi)])
	newmolecule[0][:,4:7]-=sulfur[0,4:7]
	for i in range(len(newmolecule[0])):
		newmolecule[0][i,4:7]=np.dot(rotateAxis(oldir,newdir),np.transpose(newmolecule[0][i,4:7]))
	newmolecule[0][:,4:7]+=sulfur[0,4:7]
	sulfur = newmolecule[0][(newmolecule[0][:,2]==4)]
	blankmolecule=addMolecule(blankmolecule,newmolecule)	
		

print "Minimum in x is " + str(np.amin(blankmolecule[0][:,4]))
print "Maximum in x is " + str(np.amax(blankmolecule[0][:,4]))
print "Minimum in y is " + str(np.amin(blankmolecule[0][:,5]))
print "Maximum in y is " + str(np.amax(blankmolecule[0][:,5]))
print "Minimum in y is " + str(np.amin(blankmolecule[0][:,5]))
print "Maximum in z is " + str(np.amax(blankmolecule[0][:,6]))

xyzfile = open(xyzfile,'w')
xyzfile.write(str(blankmolecule[0].shape[0])+"\n\n")
np.savetxt("temp.xyz",blankmolecule[0][:,[2,4,5,6]],'%d %f %f %f')
atomtmp = open("temp.xyz",'r')
xyzfile.write(atomtmp.read())

editFile(lammp_template,"addmolecule.lmp",blankmolecule[0],blankmolecule[1],blankmolecule[2],blankmolecule[3])
