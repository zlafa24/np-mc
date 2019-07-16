import os,sys
import time
import npmc.move_class
import npmc.simulation_class as simc

init_file = os.path.abspath('./hexo.in')
datafile = os.path.abspath('./hexo.data')
dumpfile = os.path.abspath('./hexo_out.xyz')
temp=298.15

sim = simc.Simulation(init_file,datafile,dumpfile,temp,anchortype=5,max_disp=5.0,type_lengths=(10,10),parallel=False)
print("Energy before exclude_type is "+str(sim.get_total_PE()))
sim.exclude_type(1,1)
print("Energy after exclude_type is "+str(sim.get_total_PE()))
sim.minimize()

numsteps=50

start = time.time()
for i in range(numsteps):  
    if((i%1)==0):
        sim.dump_atoms()
    print("Simulation at step "+str(i+1))
    sim.perform_mc_move()
print(time.time()-start)
