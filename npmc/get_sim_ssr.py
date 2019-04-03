import os,sys
from subprocess import call, check_output
#sys.path.insert(0,os.path.abspath('/scratch/snm8xf/np-mc/src'))

import numpy as np
import scipy.stats
import npmc.molecule_class
import npmc.maldi_class as mldc

#xyz_file = 'regrow.xyz'
#folder_prefix=sys.argv[3]
neigh_dist=6#float(sys.argv[1]) #6 (angstroms)
fragment_num=int(sys.argv[1])
numtrials=6
anchor_type=4
graph_index=0
len_typ1=13; typ1='ddt'
len_typ2=14; typ2='muda'
num_samples = 50000

cwd = os.getcwd()
os.chdir(f'{cwd}/{typ1}_{typ2}')
spectra = np.empty((numtrials,fragment_num+2))
for i in range(numtrials):
	'''
	#call(['cd','../trial'+str(i)])
	os.chdir(os.path.abspath('../'+folder_prefix+str(i+1)))
	files = check_output(['ls'])
	print(files)
	numlines = check_output(['head','-n','1',xyz_file])
	call(['tail','-n',str(int(numlines)+2),xyz_file],stdout=open('current.xyz','w'))
	call(['cp','current.xyz','../lt_files/lts/current'+str(i+1)+'.xyz'])
	os.chdir(os.path.abspath('../lt_files/lts'))
	call(['moltemplate.sh','-xyz','current'+str(i+1)+'.xyz','system.lt'])
	'''
	maldi = mldc.MALDISpectrum(os.path.abspath(f'{typ1}_{typ2}_trial{i+1}.data'),
		anchor_type=anchor_type,
		numsamples=num_samples,
		nn_distance=neigh_dist,
        graph_index=0,
		ligands_per_fragment=fragment_num,
		type_lengths=(len_typ1,len_typ2))
	fractions,bins = maldi.get_maldi_spectrum()
	ssr=maldi.get_SSR()
	spectra[i,0:fragment_num+1]=fractions
	spectra[i,fragment_num+1]=ssr
	#os.chdir(os.path.abspath('../'))
		
peak_headers = ''.join([f'peak{i}\t' for i in range(fragment_num+1)])
np.savetxt(f'ssr_L{fragment_num}_{typ1}_{typ2}.txt',spectra,header=f'{peak_headers}SSR',delimiter='\t')
