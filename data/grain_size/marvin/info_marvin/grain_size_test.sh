#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
echo "Running on ${node}"
fi
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/lib"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/bin"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/info/"
results_dir="${blazemark_dir}/results_grain_size"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8)
#thr=(1)
#num_iterations=()
num_iterations=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)
iter_lengths=(60 65 70)
chunk_sizes=(1)
# 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)
#chunk_sizes=(1)
#(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)

mkdir -p ${results_dir}
rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/grain_size_test.sh ${results_dir}/info_$node/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info

for ni in ${num_iterations[@]}
do
        echo " ${ni} iterations"
	for il in ${iter_lengths[@]}
		do
	        echo " task length ${il}"
		for c in ${chunk_sizes[@]}
		do
			if [ $c -le $ni ]
			then
	                        echo " chunk size ${c}"
				for th in ${thr[@]}
					do
					echo "${th} threads"

					export OMP_NUM_THREADS=1
	
					${hpx_bin_dir}/grain_size_test  -Ihpx.stacks.use_guard_pages=0 --num_iterations=${ni}  --hpx:threads=${th} --iter_length=${il} --chunk_size=1 ${chunk_size} --repetitions=6>>${results_dir}/${node}_grain_size_${th}_${c}_${il}_${ni}.dat
					echo "Run for ${ni} iterations, iter length of ${il}, chunk size ${c} on ${th} threads finished"			
				done
			fi
		done		
	done
done
