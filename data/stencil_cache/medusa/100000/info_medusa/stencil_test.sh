#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
echo "Running on ${node}"
fi

if [ $node = "marvin_old" ]
then
        module load clang/6.0.1
        module load boost/1.68.0-clang6-release
fi

module load clang
module load boost

saved_path=$LD_LIBRARY_PATH
blazemark_dir="/work/sshirzad/repos/Blazemark"
hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/lib"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/bin"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/info/"
results_dir="${blazemark_dir}/results_stencil"
#export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8)
grid_points=(10000000)
num_partitions=(1)
partition_sizes=(1)

rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/stencil_test.sh ${results_dir}/info_${node}/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info
cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}_main/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}/hpx_info/

for gp in ${grid_points[@]}
do
        echo " ${gp} grid points"
	for nx in $(seq 2 $gp)
		do
		if [ $((gp%nx)) -eq 0 ]
		then	
	        	echo " partition size ${nx}"
			np=$((gp/nx))
			echo " number of partitions ${np}"
			for th in ${thr[@]}
			do
				echo "${th} threads"

				export OMP_NUM_THREADS=1
	
				${hpx_bin_dir}/1d_stencil_4_parallel  --np=${np} --nx=${nx}  --hpx:threads=${th} -Ihpx.stacks.use_guard_pages=0>>${results_dir}/${node}_stencil_${th}_${nx}_${np}.dat
				echo "Run for ${np} points per partiton, ${nx} partitions, on ${th} threads finished"			
			done
		fi				
	done
done

