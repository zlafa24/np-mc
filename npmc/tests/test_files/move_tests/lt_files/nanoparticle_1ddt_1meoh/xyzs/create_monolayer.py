import numpy as np

def rot_quat(vector,theta,rot_axis):

    rot_axis = rot_axis/np.linalg.norm(rot_axis)
    vector_mag = np.linalg.norm(vector)
    quat = np.array([np.cos(theta/2),np.sin(theta/2)*rot_axis[0],np.sin(theta/2)*rot_axis[1],np.sin(theta/2)*rot_axis[2]])
    quat_inverse = np.array([np.cos(theta/2),-np.sin(theta/2)*rot_axis[0],-np.sin(theta/2)*rot_axis[1],-np.sin(theta/2)*rot_axis[2]])
    vect_quat = np.array([0,vector[0],vector[1],vector[2]])/vector_mag
    new_vector = quat_mult(quat_mult(quat,vect_quat),quat_inverse)
    return new_vector[1:]*vector_mag

def quat_mult(q1,q2):

    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    w = w1*w2-x1*x2-y1*y2-z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    result = np.array([w,x,y,z])
    return result

def align_to_vector(positions,vector):

    molecule_vector = np.mean(positions,axis=0)-positions[0]
    anchor_position = positions[0]
    axis_rotation = np.cross(molecule_vector,vector)
    angle = np.arccos(np.dot(molecule_vector/np.linalg.norm(molecule_vector),vector/np.linalg.norm(vector)))
    for i in np.arange(1,len(positions)):
        positions[i] = rot_quat((positions[i]-anchor_position),angle,axis_rotation)+anchor_position
    return positions

#Change these variables
num_shells = 9
lig_file_1 = 'meoh.xyz'
lig_file_2 = 'ddt.xyz'
#lig_file_1 = 'SC1.xyz'
#lig_file_2 = 'SC1.xyz'
num_ligands_1 = 1
num_ligands_2 = 1
output_file = 'meoh_ddt.xyz'

#These variables should take care of themselves; still requires desired NP size and next largest NP size
shells = [1,13,42,92,162,252,362,492,642,812,1002,1212,1442,1692,1962,2252,2562,2892,3242,3612,4002]
np_file = f'np_{num_shells}h.xyz'
anchor_sites_file = 'surface_sites.xyz'

np_data = np.genfromtxt(np_file,skip_header=2,dtype=None,names=['type','x','y','z'],encoding='ascii')
np_types = np_data['type']
np_pos = np_data[['x','y','z']]
anchor_sites = np.genfromtxt(anchor_sites_file,skip_header=2,usecols=np.arange(1,4))
lig_1 = np.genfromtxt(lig_file_1,skip_header=2,dtype=None,names=['type','x','y','z'],encoding='ascii')
lig_1_pos = np.genfromtxt(lig_file_1,skip_header=2,usecols=np.arange(1,4))
lig_2 = np.genfromtxt(lig_file_2,skip_header=2,dtype=None,names=['type','x','y','z'],encoding='ascii')
lig_2_pos = np.genfromtxt(lig_file_2,skip_header=2,usecols=np.arange(1,4))
atoms = len(np_types) + num_ligands_1*len(lig_1['type']) + num_ligands_2*len(lig_2['type'])
#atoms = len(np_types) + num_ligands_1 + num_ligands_2

all_idxs = np.arange(len(anchor_sites[:,0]))
chosen_idxs = []
invalid_idxs = []
for i in range(num_ligands_1+num_ligands_2):
    site = np.random.choice(all_idxs[np.in1d(all_idxs,np.array(invalid_idxs),invert=True)],1)[0]
    chosen_idxs.append(site)
    invalid_idxs = invalid_idxs + list(np.flatnonzero(np.linalg.norm(anchor_sites[site,:]-anchor_sites,axis=1) < 4.1))
sites = anchor_sites[chosen_idxs,:]

with open(output_file,'w') as output:
    output.write(f'{atoms}\n\n')
    np.savetxt(output,np_data,fmt='%s %.5f %.5f %.5f')
    for i in range(num_ligands_1):
        trans = lig_1_pos[0,:] - sites[i]
        lig = lig_1_pos - trans
        lig = 1.0*align_to_vector(lig,sites[i])
        lig_1['x'] = lig[:,0]; lig_1['y'] = lig[:,1]; lig_1['z'] = lig[:,2]
        np.savetxt(output,lig_1,fmt='%s %.5f %.5f %.5f')
    for j in range(num_ligands_2):
        trans = lig_2_pos[0,:] - sites[j+num_ligands_1]
        lig = lig_2_pos - trans
        lig = align_to_vector(lig,sites[j+num_ligands_1])
        lig_2['x'] = lig[:,0]; lig_2['y'] = lig[:,1]; lig_2['z'] = lig[:,2]
        np.savetxt(output,lig_2,fmt='%s %.5f %.5f %.5f')
