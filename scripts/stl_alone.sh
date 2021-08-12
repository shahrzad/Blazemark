#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, medusa by default"
node="medusa"
else
node=$1
echo "Running on ${node}"
fi


module load gcc 
module load boost

saved_path=$LD_LIBRARY_PATH
blazemark_dir="/work/sshirzad/repos/Blazemark"
results_dir="${blazemark_dir}/results_stl"

thr=(1 2 3 4 5 6 7 8 10 12 16 20 32 40)

rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/stl_alone.sh ${results_dir}/info_${node}/
np=500000000
for th in ${thr[@]}
do
        for rep in {1..6}
	do
                echo "run # ${rep}">>${results_dir}/${node}_stl_${th}_${np}.dat

	        TBB_NUM_THREADS=${th} /home/sshirzad/src/STLTest/build/stlTest>>${results_dir}/${node}_stl_${th}_${np}.dat
	done
done

