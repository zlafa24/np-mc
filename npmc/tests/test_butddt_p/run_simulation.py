import os,sys
import time
import npmc.move_class
import npmc.simulation_class as simc

init_file = os.path.abspath('./system.in')
datafile = os.path.abspath('./system.data')
dumpfile = os.path.abspath('./but_ddt_p.xyz')
temp=298.15

sim = simc.Simulation(init_file,datafile,dumpfile,temp,anchortype=4,max_disp=5.0,type_lengths=(5,13),parallel=True)
time.sleep(8)
print("Energy before exclude_type is "+str(sim.get_total_PE()))
sim.exclude_type(1,1)
print("Energy after exclude_type is "+str(sim.get_total_PE()))
sim.minimize()

numsteps=100

start = time.time()
for i in range(numsteps):  
    if((i%1000)==0):
        sim.dump_atoms()
    print("Simulation at step "+str(i+1))
    sim.perform_mc_move()
print(time.time()-start)
