#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
if [ $node = "marvin_old" ]
then
        module load clang/6.0.1
        module load boost/1.68.0-clang6-release
fi
echo "Running on ${node}"
fi

counter=1
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/work/sshirzad/repos/Blazemark"
hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/lib"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/bin"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/info/"
results_dir="${blazemark_dir}/results_grain_size"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8)
#thr=(8)
#num_iterations=()
num_iterations=(10000 100000 1000000 10000000)
iter_lengths=(1)

mkdir -p ${results_dir}
rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/grain_size_test_spt.sh ${results_dir}/info_${node}/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info
cp ${hpx_source_dir}/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}/hpx_info/
cp ${hpx_source_dir}/hpx/parallel/util/detail/splittable_task.hpp ${results_dir}/info_${node}/hpx_info/
cp ${hpx_source_dir}/hpx/parallel/executors/splittable_executor.hpp ${results_dir}/info_${node}/hpx_info/


for ni in ${num_iterations[@]}
do
        echo " ${ni} iterations"
	for il in ${iter_lengths[@]}
	do
	echo " task length ${il}"
		for th in ${thr[@]}
			do
			echo "${th} threads"

			export OMP_NUM_THREADS=1
			if [ $counter == 1 ]
			then
				${hpx_bin_dir}/grain_size_test --spt --counter -Ihpx.stacks.use_guard_pages=0 --num_iterations=${ni}  --hpx:threads=${th} --iter_length=${il} --repetitions=6 --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${node}_sptctr_grain_size_${th}_${c}_${il}_${ni}.dat
			else
				${hpx_bin_dir}/grain_size_test --spt  -Ihpx.stacks.use_guard_pages=0 --num_iterations=${ni}  --hpx:threads=${th} --iter_length=${il} --repetitions=6>>${results_dir}/${node}_spt_grain_size_${th}_${c}_${il}_${ni}.dat
				echo "Run for ${ni} iterations, iter length of ${il}, on ${th} threads finished"			
			fi
		done
	done
done

