import os,sys
import time
import npmc.move_class
import npmc.simulation_class as simc

init_file = os.path.abspath('./system_trappe.in')
datafile = os.path.abspath('./system_trappe.data')
dumpfile = os.path.abspath('./system_trappe.xyz')
temp=298.15

sim = simc.Simulation(init_file,datafile,dumpfile,temp,anchortype=5,max_disp=5.0,type_lengths=(10,10),parallel=False)
print("Energy before exclude_type is "+str(sim.get_total_PE()))
sim.exclude_type(1,1)
print("Energy after exclude_type is "+str(sim.get_total_PE()))
sim.minimize()

numsteps=10000

start = time.time()
energyfile = 'full_potential_energy.txt'
with open(energyfile, 'w') as file:
    for i in range(numsteps):
        etot = sim.lmp.extract_compute("pair_pe",0,0)
        evdw = sim.lmp.extract_compute("pair_total",0,0)
        edih = sim.lmp.extract_compute("mol_pe",0,0)
        eang = sim.lmp.extract_compute("ang_pe",0,0)
        ebond = sim.lmp.extract_compute("bond_pe",0,0)
        file.write('%i\t%f\t%f\t%f\t%f\t%f' % (i,etot,evdw,edih,eang,ebond))
        file.write('\n')
        if((i%1)==0):
            sim.dump_atoms()
            print("Simulation at step "+str(i+1))
        sim.perform_mc_move()
    print(time.time()-start)
