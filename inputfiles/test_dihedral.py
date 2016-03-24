#!/usr/bin/python
import numpy as np
from math import *

def quat_mult(q1,q2):
	w1,x1,y1,z1 = q1
	w2,x2,y2,z2 = q2
	w = w1*w2-x1*x2-y1*y2-z1*z2
	x = w1*x2 + x1*w2 + y1*z2 - z1*y2
	y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    	z = w1*z2 + z1*w2 + x1*y2 - y1*x2
	return np.array([w,x,y,z])

def rot_quat(vector,theta,rot_axis):
	rot_axis = rot_axis/np.linalg.norm(rot_axis)
	vector_mag = np.linalg.norm(vector)
	quat = np.array([cos(theta/2),sin(theta/2)*rot_axis[0],sin(theta/2)*rot_axis[1],sin(theta/2)*rot_axis[2]])
	quat_inverse = np.array([cos(theta/2),-sin(theta/2)*rot_axis[0],-sin(theta/2)*rot_axis[1],-sin(theta/2)*rot_axis[2]])
	quat = quat/np.linalg.norm(quat)
	quat_inverse = quat_inverse/(np.linalg.norm(quat_inverse)**2)
	
	vect_quat = np.array([0,vector[0],vector[1],vector[2]])/vector_mag
	new_vector = quat_mult(quat_mult(quat,vect_quat),quat_inverse)
	return new_vector[1:]*vector_mag

def rotate_dihedral_quat(dih_atoms,angle,atoms2rotate):
	rot_angle = angle-calc_dih_angle(dih_atoms)
	#rot_angle = angle
	rot_axis = dih_atoms[2,4:7]-dih_atoms[1,4:7]
	for atom in atoms2rotate:
		atom[4:7] = rot_quat((atom[4:7]-dih_atoms[2,4:7]),rot_angle,rot_axis)+dih_atoms[2,4:7]
	return atoms2rotate
		

def calc_dih_angle(dih_atoms):
	b1 = dih_atoms[1,4:7]-dih_atoms[0,4:7]
	b2 = dih_atoms[2,4:7]-dih_atoms[1,4:7]
	b3 = dih_atoms[3,4:7]-dih_atoms[2,4:7]
	#b4 = np.cross(b1,b2)
	#b5 = np.cross(b2,b4)
	b2norm = b2/np.linalg.norm(b2)
	n1 = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
	n2 = np.cross(b2,b3)/np.linalg.norm(np.cross(b2,b3))
	m1 = np.cross(n1,b2norm)
	angle = atan2(np.dot(m1,n2),np.dot(n1,n2))
	angle=((angle-pi)*(-1)+2*pi)%(2*pi)
	return angle
	#angle =  atan2(np.dot(b3,b4),np.dot(b3,b5)*sqrt(np.dot(b2,b2)))
	#return (angle if angle>=0 else angle+2*pi)

def rotate_dihedral(dih_atoms,angle,atoms2rotate):
	b2 = dih_atoms[2,4:7]-dih_atoms[1,4:7]
	print "\nRotation axis is "+str(b2)+"\n"
	b3 = dih_atoms[3,4:7]-dih_atoms[2,4:7]
	rot_axis = b2/np.linalg.norm(b2)
	init_vector = b3
	#n1 = np.cross((dih_atoms[1,4:7]-dih_atoms[0,4:7]),rot_axis)
	#n2 = np.cross((dih_atoms[3,4:7]-dih_atoms[2,4:7]),rot_axis)
	#init_angle = np.arccos(np.dot((n1/np.linalg.norm(n1)),(n2/np.linalg.norm(n2))))
	init_angle = calc_dih_angle(dih_atoms)
	#rot_angle = angle-init_angle
	rot_angle = angle
	#print "Initial angle is "+str(init_angle)+" desired angle is "+str(angle)+" therefore rotating "+str(rot_angle)
	skewmat = np.array([[0,-rot_axis[2],rot_axis[1]],[rot_axis[2],0,-rot_axis[0]],[-rot_axis[1],rot_axis[0],0]])
	print "\nSkew matrix is "+str(skewmat)+"\n"
	rot_matrix = np.identity(3)+sin(rot_angle)*skewmat+(2*sin(rot_angle/2)**2)*np.linalg.matrix_power(skewmat,2)
	print "\nRotation Matrix is "+str(rot_matrix)+"\n"
	#print "\nDihedral atoms looks like this "+str(dih_atoms[:,4:7])+"\n"
	for atom in atoms2rotate:	
		atom[4:7] = atom[4:7] - dih_atoms[2,4:7]
		print "\nAtom transposed is "+str(np.transpose(atom[4:7]))+"\n"
		atom[4:7] = np.transpose(np.dot(rot_matrix,np.transpose(atom[4:7])))+dih_atoms[2,4:7]
	#print "\nNow dihedral atoms looks like this "+str(dih_atoms[:,4:7])+"\n"
	return atoms2rotate

if __name__=='__main__':
	dih_atoms = np.arange(28,dtype=float).reshape((4,7))
	dih_atoms[0,4:7] = np.array([0.0,0.0,0.0],dtype=float)
	dih_atoms[1,4:7] = np.array([0.,1.,0.],dtype=float)
	dih_atoms[2,4:7] = np.array([1.5,1.0,0.],dtype=float)
	dih_atoms[3,4:7] = np.array([1.5,2.,0.],dtype=float)
	print "Initial dihedral angle is "+str(calc_dih_angle(dih_atoms)/pi)+"\n"
	atoms2rotate = np.arange(14,dtype=float).reshape((2,7)).astype(float)
	#print "dih_atoms are "+str(dih_atoms)
	atoms2rotate[0,4:7] = dih_atoms[3,4:7]
	#print atoms2rotate
	atoms2rotate = rotate_dihedral_quat(dih_atoms,3*pi/2,atoms2rotate)
	print "Rotate dihedral returns "+str(atoms2rotate)+"\n from original dihedrals "+str(dih_atoms)
	dih_atoms[3,4:7] = atoms2rotate[0,4:7]
	print "Angle is now "+str(calc_dih_angle(dih_atoms)/pi)
	#dih_atoms[3,4:7] = np.array([1.,1.,1.])
	atoms2rotate = rotate_dihedral_quat(dih_atoms,3*pi/2,atoms2rotate)
	print "Rotate dihedral returns "+str(atoms2rotate)+"\n from original dihedrals "+str(dih_atoms)
	dih_atoms[3,4:7] = atoms2rotate[0,4:7]
        print "Angle is now "+str(calc_dih_angle(dih_atoms)/pi)
	#dih_atoms[3,4:7] = np.array([1.,0.,0.])
        atoms2rotate = rotate_dihedral_quat(dih_atoms,3*pi/2,atoms2rotate)
	print "Rotate dihedral returns "+str(atoms2rotate)+"\n from original dihedrals "+str(dih_atoms)
	dih_atoms[3,4:7] = atoms2rotate[0,4:7]
	print "Angle is now "+str(calc_dih_angle(dih_atoms)/pi)

