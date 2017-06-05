#!/usr/bin/python
import os,sys
sys.path.insert(0,os.path.expanduser('~/np-mc/src'))
import simulation_class as simc

init_file = os.path.abspath('./system.in')
datafile = os.path.abspath('./system.data')
dumpfile = os.path.abspath('./regrow.xyz')
temp=298.15

sim = simc.Simulation(init_file,datafile,dumpfile,temp)
sim.minimize()

numsteps=10000

for i in range(numsteps):
    if (i+1)%100==0:
        print("Simulation at step "+str(i+1))
    sim.perform_mc_move()
