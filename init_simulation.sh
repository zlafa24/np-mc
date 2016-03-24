#!/bin/bash
r=2
dest_path=$(pwd)
src_path=$(dirname $0)
grafting_density=$1
configuration=$2
number=$(echo "${grafting_density}*4*3.14159*${r}^2/1"|bc)
echo "Grafting ${number} ligands onto the np surface"
cp -v ${src_path}/inputfiles/* ./
cd ${src_path}
mc_routine_version=$(git log -n 1|grep Date)
./make_lammp_input.py -n ${number} -f relaxed_ico.xyz -c ${configuration}
cd ${dest_path}
mv ${src_path}/addmolecule.lmp ./addmolecule_184_rand.lmp
#echo "Simulation started on $(date)\nWith mc_routine version: ${mc_routine_version}\nGrafting Density: $1\nStarting Configuration: $2">README
