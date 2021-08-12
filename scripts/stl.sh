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
hpx_dir="/home/sshirzad/src/hpx/build_release_gcc_no_hpxmp_${node}_main/lib"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_gcc_no_hpxmp_${node}_main/bin"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_gcc_no_hpxmp_${node}_main/info/"
results_dir="${blazemark_dir}/results_stl"
#export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-gcc6.0.1/release/lib:$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8 10 12 16 20 32 40)
#thr=(5 6 7 8)

rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/stl.sh ${results_dir}/info_${node}/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info
np=500000000
for th in ${thr[@]}
do
       for rep in {1..6}
	do
                echo "run # ${rep}">>${results_dir}/${node}_stl_${th}_${np}.dat

               OMP_NUM_THREADS=1 ${hpx_bin_dir}/parallel_algorithms_test  --hpx:threads=${th}>>${results_dir}/${node}_hpx_${th}_${np}.dat
	done
done

