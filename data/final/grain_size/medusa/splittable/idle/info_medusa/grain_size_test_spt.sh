#!/bin/bash
if [ $# -eq 0 ]
then
	echo "node not specified, marvin by default"
	node="marvin"
	echo "split type= all by default"
	split_type="all"
	min_task_size=0
elif [ $# -eq 1 ]
then
	node=$1
        echo "split type= all by default"
        split_type="all"
	min_task_size=0

elif [ $# -eq 2 ]
then
        node=$1
        echo "min task size=0 by default"
        split_type=$2
	min_task_size=0
else
	node=$1
	split_type=$2
	min_task_size=$3
fi

hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/lib"

if [ $node = "marvin_old" ]
then
        module load clang/6.0.1
        module load boost/1.68.0-clang6-release
	export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
fi
echo "Running on ${node} split type ${split_type}"


counter=0
#split_type="idle"
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/work/sshirzad/repos/Blazemark"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/bin"
hpx_source_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}_main/include"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/info/"
results_dir="${blazemark_dir}/results_grain_size"

thr=(1 2 3 4 5 6 7 8)
#thr=(8)
#num_iterations=()
num_iterations=(1000 10000 100000 1000000 10000000 100000000)
#num_iterations=(1 10 100 1000 10000 100000)
iter_lengths=(1)

mkdir -p ${results_dir}
#rm -rf ${results_dir}/*.dat
#rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/grain_size_test_spt.sh ${results_dir}/info_${node}/
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
		for th in ${thr[@]}
			do
			echo "${th} threads"

			export OMP_NUM_THREADS=1
			if [ $counter == 1 ]
			then
				${hpx_bin_dir}/grain_size_test --spt --counter -Ihpx.stacks.use_guard_pages=0 --split_type=${split_type} --min_task_size=${min_task_size} --chunk_size=1 --num_iterations=${ni}  --hpx:threads=${th} --iter_length=${il} --repetitions=3 --hpx:print-counter=/threads/time/cumulative --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead --hpx:ini=hpx.thread_queue.min_tasks_to_steal_staged=0>>${results_dir}/${node}-sptctr_${split_type}_grain_size_${th}_${min_task_size}_${il}_${ni}.dat
			else
				${hpx_bin_dir}/grain_size_test --spt --min_task_size=${min_task_size} --split_type=${split_type}  -Ihpx.stacks.use_guard_pages=0 --chunk_size=1 --num_iterations=${ni}  --hpx:threads=${th} --iter_length=${il} --repetitions=6 --hpx:ini=hpx.thread_queue.min_tasks_to_steal_staged=0>>${results_dir}/${node}-spt_${split_type}_grain_size_${th}_${min_task_size}_${il}_${ni}.dat
				
				echo "Run for ${ni} iterations, iter length of ${il}, on ${th} threads finished"			
			fi
		done
	done
done

