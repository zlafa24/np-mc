#!/usr/bin/python
import numpy as np
import string
from math import *
import random as rnd


from libc.stdio cimport *
from libc.string cimport *

cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    ssize_t getline(char **, size_t *, FILE *)

cdef extern from "string.h":
    int strcmp(const char *, const char *)
    char *strstr(const char *, const char *)

def readfile_c(filename):
    print "Starting to read file "+str(filename)
    filename_bytes = filename.encode("UTF-8")
    cdef char* fname = filename_bytes
    cdef FILE* inputfile
    print "Opening file"
    inputfile = fopen(fname,"rb")
    if inputfile == NULL:
        print "No such file: '%s'" % filename
    currentId=0
    startPos=1
    
    cdef char* line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    cdef char* atomstring = "atoms"
    cdef char* atomheader = "Atoms"

    print "Finding numatms"
    read = getline(&line,&l,inputfile)
    print "First line read"
    while(strstr(line,atomstring)==NULL):
        print "Reading line "+str(line)
        read = getline(&line,&l,inputfile)
        startPos+=1
    atomline = str(line)
    numwords = atomline.split()
    numatms = int(numwords[0])
    atoms = np.zeros((numatms,7))
    print "Finding atoms"
    while(strstr(line,atomheader)==NULL):
        print "Reading line "+str(line)
        read = getline(&line,&l,inputfile)
        startPos+=1
        continue
    read = getline(&line,&l,inputfile)
    startPos+=1
    for j in range(numatms):
        read = getline(&line,&l,inputfile)
        atomline=line
        record = atomline.split()
        atoms[j,0]=int(record[0])
        atoms[j,1]=int(record[1])
        atoms[j,2]=int(record[2])
        atoms[j,3]=float(record[3])
        atoms[j,4]=float(record[4])
        atoms[j,5]=float(record[5])
        atoms[j,6]=float(record[6])
    fclose(inputfile)
    return atoms

def editFile_c(filein,atoms,fileout):
    filein_bytes = filein.encode("UTF-8")
    fileout_bytes = fileout.encode("UTF-8")
    cdef char* fname_in = filein_bytes
    cdef char* fname_out = fileout_bytes
    cdef FILE* inputfile = fopen(fname_in,"rb")
    cdef FILE* outputfile = fopen(fname_out,"wb")

    cdef char* line = NULL
    cdef size_t l = 0
    cdef ssize_t read
    cdef char* atomstring = "atoms"
    cdef char* atomheader = "Atoms"
    cdef char* newline = "\n"

    currentId=0
    read = getline(&line,&l,inputfile)
    fputs(line,outputfile)
    while(strstr(line,atomstring)==NULL):
        read = getline(&line,&l,inputfile)
        fputs(line,outputfile)
    while(strstr(line,atomheader)==NULL):
        read = getline(&line,&l,inputfile)
        fputs(line,outputfile)
    read = getline(&line,&l,inputfile)
    fputs(line,outputfile)
    cdef int atoms0
    cdef int atoms1
    cdef int atoms2
    cdef float atoms3
    cdef float atoms4
    cdef float atoms5
    cdef float atoms6
    for i in xrange(atoms.shape[0]):
        atoms0=atoms[i,0]
        atoms1=atoms[i,1]
        atoms2=atoms[i,2]
        atoms3=atoms[i,3]
        atoms4=atoms[i,4]
        atoms5=atoms[i,5]
        atoms6=atoms[i,6]
        fprintf(outputfile,"%7i %7d %4d %10.6f %16.8f %16.8f %16.8f\n",atoms0,atoms1,atoms2,atoms3,atoms4,atoms5,atoms6)
        read = getline(&line,&l,inputfile)
    read = getline(&line,&l,inputfile)
    while(read != -1):
        fputs(line,outputfile)
        read = getline(&line,&l,inputfile)
    fclose(inputfile)
    fclose(outputfile)


def readfile(filename):
    input = open(filename,"r",8192)
    print "reading file " + filename
    currentId=0
    startPos=1
    line = input.readline()
    out = line.find("atoms")
    while(out==-1):
        line = input.readline()
        out = line.find("atoms")
        startPos+=1
    numwords = line.split()
    numatms = int(numwords[0])
    atoms = np.zeros((numatms,7))
    while(input.readline().find("Atoms")==-1):
        startPos+=1
        continue
    input.readline()
    startPos+=1
    for j in range(numatms):
        line=input.readline()
        record = line.split()
        atoms[j,0]=int(record[0])
        atoms[j,1]=int(record[1])
        atoms[j,2]=int(record[2])
        atoms[j,3]=float(record[3])
        atoms[j,4]=float(record[4])
        atoms[j,5]=float(record[5])
        atoms[j,6]=float(record[6])
    input.close()
    return atoms

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
    R = np.identity(3) + vX +np.linalg.matrix_power(vX,2)*((1-c)/(s**2))
    return R

def randomRotate(atoms,molId,max_angle):
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
        print "Rotating "+str(angle)+"radians about radial axis"
    else:
        angle = rnd.uniform(-max_angle,max_angle)
        print "Rotating "+str(angle)+"radians about theta or phi axis"
    rotMatrix = np.array([cos(angle)+(rotAxis[0]**2)*(1-cos(angle)),rotAxis[0]*rotAxis[1]*(1-cos(angle))-rotAxis[2]*sin(angle),rotAxis[0]*rotAxis[2]*(1-cos(angle))+rotAxis[1]*sin(angle),rotAxis[0]*rotAxis[1]*(1-cos(angle))+rotAxis[2]*sin(angle),cos(angle)+(rotAxis[1]**2)*(1-cos(angle)),rotAxis[1]*rotAxis[2]*(1-cos(angle))-rotAxis[0]*sin(angle),rotAxis[0]*rotAxis[2]*(1-cos(angle))-rotAxis[1]*sin(angle),rotAxis[1]*rotAxis[2]*(1-cos(angle))+rotAxis[0]*sin(angle),cos(angle)+(rotAxis[2]**2)*(1-cos(angle))])
    rotMatrix = np.reshape(rotMatrix,(3,3))
    atoms[indices,4:7]-=sulfur[0,4:7]
    for i in xrange(indices.shape[0]):
        atoms[indices[i],4:7]=np.dot(rotMatrix,np.transpose(atoms[indices[i],4:7]))
    atoms[indices,4:7]+=sulfur[0,4:7]

def swapMolecules(molId1,molId2,atoms,centerRotation,rotation):
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
    if rotation is "align":
        a1old = a1
        a1new = b1
        a2old = a2
        a2new = b2
    elif rotation is "swap":
        a1old = a1
        a1new = a2
        a2old = a2
        a2new = a1
    else:
        a1old = b2
        a1new = b1
        a2old = b1
        a2new = b2
    atoms[atoms1,4:7]-=sulfur2[0,4:7]
    for i in range(len(atoms1)):
        atoms[atoms1[i],4:7]=np.dot(rotateAxis(a1old,a1new),np.transpose(atoms[atoms1[i],[4,5,6]]))
    atoms[atoms1,4:7]+=sulfur2[0,4:7]
    atoms[atoms2,4:7]-=sulfur1[0,4:7]
    for i in range(len(atoms2)):
        atoms[atoms2[i],4:7]=np.dot(rotateAxis(a2old,a2new),np.transpose(atoms[atoms2[i],[4,5,6]]))
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
