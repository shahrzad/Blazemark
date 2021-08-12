#!/bin/bash
module load clang/10.0.0
module load boost/1.72.0-release
if [ $# -eq 0 ]
then
        echo "node not specified, marvin by default"
        node="marvin"
else
        node=$1
fi

hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main/lib"

echo "Running on ${node}"


counter=0
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/work/sshirzad/repos/Blazemark"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_build_dir="${hpx_source_dir}/build_release_clang_no_hpxmp_${node}_main"
hpx_bin_dir="${hpx_build_dir}/bin"
hpx_log_dir="${hpx_build_dir}/info/"
results_dir="${blazemark_dir}/results_grain_size"

array_sizes_log=(6)
thr=(1 2 3 4 5 6 7 8 10 12 16)
chunk_sizes=(1 2 3 4 5 6 7 8 9 10)

mkdir -p ${results_dir}
rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/grain_size_benchmarks_updated.sh ${results_dir}/info_${node}/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info
cp ${hpx_source_dir}/libs/parallelism/algorithms/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}/hpx_info/

for asl in ${array_sizes_log[@]}
do
	as=$((10**asl))
	for b in {1..10}
	do
		for c in $(seq 0 $((asl-1)))
		do
			chunk_size=$((b*10**c))
			echo "chunk size ${chunk_size}"
			for th in ${thr[@]}
				do
				echo "${th} threads"
			
				export OMP_NUM_THREADS=1
			
			                        echo $b $c "chunk size ${chunk_size}" $th
${hpx_bin_dir}/grain_size_benchmarks_test  -Ihpx.stacks.use_guard_pages=0 --array_size=${as} --hpx:threads=${th} --chunk_size=${c} --repetitions=1
				perf stat -e task-clock,cycles,instructions,cache-references,cache-misses ${hpx_bin_dir}/grain_size_benchmarks_test  -Ihpx.stacks.use_guard_pages=0 --array_size=${as} --hpx:threads=${th} --chunk_size=${c} --repetitions=1&>>${results_dir}/${node}_grain_size_benchmark_${th}_${chunk_size}_${as}.dat
				echo "Run for array size ${as} iterations, chunk size ${c} on ${th} threads finished"			
			done
		done		
	done
done

