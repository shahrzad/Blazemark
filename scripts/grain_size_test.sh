#!/bin/bash
if [ $# -eq 0 ]
then
        echo "node not specified, marvin by default"
        node="marvin"
else
        node=$1
fi

hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/lib"

if [ $node = "marvin_old" ]
then
        module load clang/6.0.1
        module load boost/1.68.0-clang6-release
        export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
fi
echo "Running on ${node}"


counter=0
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/work/sshirzad/repos/Blazemark"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/bin"
hpx_source_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}_main/include"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/info/"
results_dir="${blazemark_dir}/results_grain_size"

thr=(1 2 3 4 5 6 7 8 10 12 16)
num_iterations=(1000 10000 100000 1000000 10000000)
num_iterations=(100000)
iter_lengths=(1)
chunk_sizes=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 200000 300000 400000 500000 1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000)

mkdir -p ${results_dir}
rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/grain_size_test.sh ${results_dir}/info_${node}/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info
cp ${hpx_source_dir}/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}/hpx_info/
cp ${hpx_source_dir}/hpx/parallel/util/detail/splittable_task.hpp ${results_dir}/info_${node}/hpx_info/
cp ${hpx_source_dir}/hpx/parallel/executors/splittable_executor.hpp ${results_dir}/info_${node}/hpx_info/
cp ${hpx_source_dir}/hpx/executors/splittable_executor.hpp ${results_dir}/info_${node}/hpx_info
cp ${hpx_source_dir}/hpx/executors/detail/splittable_task.hpp ${results_dir}/info_${node}/hpx_info


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
	
					${hpx_bin_dir}/grain_size_test  -Ihpx.stacks.use_guard_pages=0 --num_iterations=${ni}  --hpx:threads=${th} --iter_length=${il} --chunk_size=${c} --repetitions=6>>${results_dir}/${node}_grain_size_${th}_${c}_${il}_${ni}.dat
					echo "Run for ${ni} iterations, iter length of ${il}, chunk size ${c} on ${th} threads finished"			
				done
			fi
		done		
	done
done

