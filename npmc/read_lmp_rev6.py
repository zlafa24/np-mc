#!/usr/bin/python
import sys
import numpy as np
import string
from math import *
import random as rnd
from subprocess import call
import itertools as itt

def readAtoms(filename):
    input = open(filename,"r")
    print("reading file {}".format(filename))
    currentId=0
    line = input.readline()
    out = line.find("atoms")
    while(out==-1):
        line = input.readline()
        out = line.find("atoms")
    print(line)
    numwords = line.split()
    numatms = int(numwords[0])
    atoms = np.zeros((numatms,7))
    while(input.readline().find("Atoms")==-1):
        continue
    input.readline()
    line=input.readline
    for j in range(numatms):
        line=input.readline()
        record = line.split()
        for i in range(7):
            if i<3:
                atoms[j,i]=int(record[i])
            else:
                atoms[j,i]=float(record[i])
    return atoms

def readBonds(filename):
    input = open(filename,"r")
    print("reading file {}".format(filename))
    currentId=0
    line = input.readline()
    out = line.find("bonds")
    while(out==-1):
        line = input.readline()
        out = line.find("bonds")
    print(line)
    numwords = line.split()
    numbonds = int(numwords[0])
    bonds = np.zeros((numbonds,4))
    while(input.readline().find("Bonds")==-1):
        continue
    input.readline()
    line=input.readline
    for j in range(numbonds):
        line=input.readline()
        record = line.split()
        for i in range(4):
            bonds[j,i]=int(record[i])
    return bonds

def readAngles(filename):
    input = open(filename,"r")
    print("reading file ".format(filename))
    currentId=0
    line = input.readline()
    out = line.find("angles")
    while(out==-1):
        line = input.readline()
        out = line.find("angles")
    print(line)
    numwords = line.split()
    numangles = int(numwords[0])
    angles = np.zeros((numangles,5))
    while(input.readline().find("Angles")==-1):
        continue
    input.readline()
    line=input.readline
    for j in range(numangles):
        line=input.readline()
        record = line.split()
        for i in range(5):
            angles[j,i]=int(record[i])
    return angles

def readDihedrals(filename):
    input = open(filename,"r")
    print("reading file ".format(filename))
    currentId=0
    line = input.readline()
    out = line.find("dihedrals")
    while(out==-1):
        line = input.readline()
        out = line.find("dihedrals")
    print(line)
    numwords = line.split()
    numdiheds = int(numwords[0])
    diheds = np.zeros((numdiheds,6))
    while(input.readline().find("Dihedrals")==-1):
        continue
    input.readline()
    line=input.readline
    for j in range(numdiheds):
        line=input.readline()
        record = line.split()
        for i in range(6):
            diheds[j,i]=int(record[i])
    return diheds

def readAll(inputfile):
    atoms=readAtoms(inputfile)
    bonds=readBonds(inputfile)
    angles=readAngles(inputfile)
    diheds=readDihedrals(inputfile)
    return (atoms,bonds,angles,diheds)

def deleteMolecule(atoms,bonds,angles,diheds,molId):
    molatoms=atoms[atoms[:,1]==molId][:,0]
    molbonds = np.array([])
    molangles = np.array([])
    moldiheds = np.array([])
    for atom in molatoms:
        molbonds = np.union1d(molbonds,np.where(((bonds[:,2]==atom)|(bonds[:,3]==atom)))[0])
        molangles = np.union1d(molangles,np.where(((angles[:,2]==atom)|(angles[:,3]==atom)|(angles[:,4]==atom)))[0])
        moldiheds = np.union1d(moldiheds,np.where(((diheds[:,2]==atom)|(diheds[:,3]==atom)|(diheds[:,4]==atom)|(diheds[:,5]==atom)))[0])
    newatoms=np.delete(atoms,np.where((atoms[:,1]==molId))[0],0)
    newbonds=np.delete(bonds,molbonds,0)
    newangles=np.delete(angles,molangles,0)
    newdiheds=np.delete(diheds,moldiheds,0)
    return (newatoms,newbonds,newangles,newdiheds)

def addMolecule(molecules,newmolecule):
    atoms=molecules[0]
    bonds=molecules[1]
    angles=molecules[2]
    diheds=molecules[3]
    replacelist=[]
    lastAtmId = np.amax(atoms[:,0]) if atoms.size>0 else 0
    lastMolId = np.amax(atoms[:,1]) if atoms.size>0 else 0
    lastBondId = np.amax(bonds[:,0]) if bonds.size>0 else 0
    lastAngleId = np.amax(angles[:,0]) if angles.size>0 else 0
    lastDihedId = np.amax(diheds[:,0]) if diheds.size>0 else 0
    for i in range(len(newmolecule[0][:,0])):
        oldid=newmolecule[0][i,0]
        newmolecule[0][i,0]=lastAtmId+i+1
        replacelist.append((oldid,newmolecule[0][i,0]))
        newmolecule[0][i,1]=lastMolId+1
    for i in range(len(newmolecule[1][:,0])):
        newmolecule[1][i,0]=lastBondId+i+1
        if(newmolecule[1][i,2] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[1][i,2]]
            newmolecule[1][i,2]=newid[0]
        if(newmolecule[1][i,3] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[1][i,3]]
            newmolecule[1][i,3]=newid[0]
    for i in range(len(newmolecule[2][:,0])):
        newmolecule[2][i,0]=lastAngleId+i+1
        if(newmolecule[2][i,2] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[2][i,2]]
            newmolecule[2][i,2]=newid[0]
        if(newmolecule[2][i,3] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[2][i,3]]
            newmolecule[2][i,3]=newid[0]
        if(newmolecule[2][i,4] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[2][i,4]]
            newmolecule[2][i,4]=newid[0]
    for i in range(len(newmolecule[3][:,0])):
        newmolecule[3][i,0]=lastDihedId+i+1
        if(newmolecule[3][i,2] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[3][i,2]]
            newmolecule[3][i,2]=newid[0]
        if(newmolecule[3][i,3] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[3][i,3]]
            newmolecule[3][i,3]=newid[0]
        if(newmolecule[3][i,4] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[3][i,4]]
            newmolecule[3][i,4]=newid[0]
        if(newmolecule[3][i,5] in [x[0] for x in replacelist]):
            newid = [x[1] for x in replacelist if x[0]==newmolecule[3][i,5]]
            newmolecule[3][i,5]=newid[0]
    atoms=np.append(atoms,newmolecule[0],0)
    bonds=np.append(bonds,newmolecule[1],0)
    angles=np.append(angles,newmolecule[2],0)
    diheds=np.append(diheds,newmolecule[3],0)
    return (atoms,bonds,angles,diheds)

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

def shiftMolecule(atoms,indices,shiftx,shifty,shiftz):
    numatms=len(indices)
    for i in range(numatms):
        atoms[indices[i],4]+=shiftx
        atoms[indices[i],5]+=shifty
        atoms[indices[i],6]+=shiftz

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
        x+=atommass*atoms[indices[i],4]
        y+=atommass*atoms[indices[i],5]
        z+=atommass*atoms[indices[i],6]
        totalmass+=atommass
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
    print("normalization is ".format((1-c)/(s**2)))
    R = np.identity(3) + vX +np.linalg.matrix_power(vX,2)*((1-c)/(s**2))
    return R
    
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

def editFile(filein,fileout,atoms,bonds,angles,diheds):
    input = open(filein,"r")
    output = open(fileout,"w")
    currentId=0
    line = input.readline()
    out = line.find("atoms")
    while(out==-1):
        output.write(line)
        line = input.readline()
        out = line.find("atoms")
    output.write("{0:12d} {1}\n".format(len(atoms),"atoms"))
    output.write("{0:12d} {1}\n".format(len(bonds),"bonds"))
    output.write("{0:12d} {1}\n".format(len(angles),"angles"))
    output.write("{0:12d} {1}\n\n".format(len(diheds),"dihedrals"))
    output.write("{0:12d} {1}\n".format(len(np.unique(atoms[:,2])),"atom types"))
    output.write("{0:12d} {1}\n".format(len(np.unique(bonds[:,1])),"bond types"))
    output.write("{0:12d} {1}\n".format(len(np.unique(angles[:,1])),"angle types"))
    output.write("{0:12d} {1}\n\n".format(len(np.unique(diheds[:,1])),"dihedral types"))
    out = line.find("xlo")
    while(out==-1):
        line = input.readline()
        out=line.find("xlo")
    output.write(line)
    while(line.find("Atoms")==-1):
        line = input.readline()
        output.write(line)
    output.write("\n")
    np.savetxt("atoms.temp",atoms,"%7d %7d %4d %10.6f %16.8f %16.8f %16.8f")
    np.savetxt("bonds.temp",bonds,"%6d %3d %6d %6d")
    np.savetxt("angles.temp",angles,"%6d %3d %6d %6d %6d")
    np.savetxt("diheds.temp",diheds,"%6d %3d %6d %6d %6d %6d") 
    atomtmp = open("atoms.temp","r")
    bondtmp = open("bonds.temp","r")
    angletmp = open("angles.temp","r")
    dihedtmp = open("diheds.temp","r")
    output.write(atomtmp.read())
    #while(input.readline().find("Bonds")==-1):
    #    continue
    output.write("\nBonds\n\n")
    output.write(bondtmp.read())
    output.write("\nAngles\n\n")
    output.write(angletmp.read())
    output.write("\nDihedrals\n\n")
    output.write(dihedtmp.read())
    #while(line != ''):
    #    line = input.readline()
    #    output.write(line)

def molCollide(atoms,indices):
    numatms = len(indices)
    xcenter = 40.9
    ycenter = 40.9
    zcenter = 40.9
    for i in range(numatms):
        x = atoms[indices[i],4]
        y = atoms[indices[i],5]
        z = atoms[indices[i],6]
        if sqrt((x-xcenter)**2+(y-ycenter)**2+(z-zcenter)**2)<20.45:
            return True
    return False

def getPotential(filename):
    pefile = open(filename,"r")
    lines=pefile.readlines()
    return lines[len(lines)-1].split()[1]

def getBondedAtoms(bonds,atomID):
    bonded1 = bonds[(bonds[:,2]==atomID)][:,3] if bonds[(bonds[:,2]==atomID)].shape[0]>0 else []
    bonded2 = bonds[(bonds[:,3]==atomID)][:,2] if bonds[(bonds[:,3]==atomID)].shape[0]>0 else []
    return np.append(np.ravel(bonded1),np.ravel(bonded2))

def getMoleculeAtoms(bonds,startID):
    atomIDs = np.empty([1])
    atomIDs[0] = startID
    bondedAtoms = getBondedAtoms(bonds,startID)
    actAtoms = [atom for atom in bondedAtoms if ((atom>0) and (not (atom in atomIDs)))]
    while(len(actAtoms)>0):
        atomIDs = np.append(atomIDs,actAtoms[0])
        bondedAtoms = getBondedAtoms(bonds,actAtoms[0])
        actAtoms = [atom for atom in bondedAtoms if ((atom>0) and (not (atom in atomIDs)))]
    return atomIDs

if __name__ == "__main__":
    inputfile=sys.argv[1]
    basename=inputfile.split('.')[0]
    atoms = readfile(inputfile)
    kb=0.0019872041 #in kcal/mol-K
    T=360
    beta = 1/(kb*T)
    molIds = atoms[atoms[:,2]==4][:,1]
    centerRotation=np.array([40.9,40.9,40.9])
    numsteps=10
    rnd.seed()
    collisions=0
    lastenergy=getPotential("pe.out")
    for i in xrange(numsteps):
        swappedList = rnd.sample(molIds,2)
        swapMolecules(swappedList[0],swappedList[1],atoms,centerRotation)
        if molCollide(atoms,np.where((atoms[:,1]==swappedList[0]))[0]) or  molCollide(atoms,np.where((atoms[:,1]==swappedList[1]))[0]):
            continue
        editFile(inputfile,atoms,basename+str(i)+".lmp")
        call(["cp",basename+str(i)+".lmp",basename+".temp"])        
        call("./runlammps.sh")
        energy=getPotential("pe.out")
        dU = energy-lastenergy
        if(dU<=0):
            atoms=readfile(basename+str(i)+".lmp")
        elif(rnd.random()<exp(-beta*dU)):
            atoms=readfile(basename+str(i)+".lmp")
