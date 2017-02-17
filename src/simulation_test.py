from simulation_class import *

init_file = "/home/slow89/np-mc/lt_files/nanoparticle/system.in"
data_file = "/home/slow89/np-mc/lt_files/nanoparticle/system.data"
dumpfile =  "/home/slow89/np-mc/lt_files/nanoparticle/regrow.xyz"
temp = 298.15

sim = Simulation(init_file,data_file,dumpfile,temp)

#sim.minimize()

