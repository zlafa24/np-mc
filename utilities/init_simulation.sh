#!/bin/bash
r=2
dest_path=$(pwd)
#grafting_density=$1
#configuration=$2
#solvent="explicit"
echo "Here"
while getopts "g:c:s:" opt; do
    echo "Here again"
    echo "Flag is ${opt}"
    case $opt in
	g)
	    grafting_density=$OPTARG
	    echo "Grafting density is now ${OPTARG}"
	    ;;
	c)
	    configuration=$OPTARG
	    #echo "Configuration is set to ${configuration}"
	    ;;
	s)
	    solvent=$OPTARG
	    ;;
    esac
done
shift $(( OPTIND -1 ))

echo -n "Enter any additional comments about this simulation:"
read comments

src_path=$(dirname $0)
#echo "Grafting density is now ${configuration}"
number=$(echo "${grafting_density}*4*3.14159*${r}^2/1"|bc)
echo "Grafting ${number} ligands onto the np surface"
cp -v ${src_path}/inputfiles/* ./

if [ "$solvent" == "explicit" ]; then
	cp ${src_path}/explicit.lmi ./in.lmi
else
	cp ${src_path}/implicit.lmi ./in.lmi
fi

cd ${src_path}
mc_routine_version=$(git log -n 1|grep Date)

./make_lammp_input.py -n ${number} -f relaxed_ico.xyz -c ${configuration}
./addcharge_rev2.py addmolecule.lmp
cp charged.lmp addmolecule.lmp
cd ${dest_path}
mv ${src_path}/addmolecule.lmp ./addmolecule_184_rand.lmp

echo "Simulation started on $(date)" > README
echo "With mc_routine version: ${mc_routine_version}">>README
echo "Grafting Density: ${grafting_density}">>README
echo "Starting Configuration: ${configuration}">>README
echo "Solvent method is: ${solvent}">>README
echo -e "\nAdditional Comments:\n${comments}">>README


#echo "Simulation started on $(date) \nWith mc_routine version: ${mc_routine_version} \n Grafting Density: $1 \n Starting Configuration: $2">README
