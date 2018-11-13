#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt"
results_dir="${blaze_dir}/blazemark/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blaze_dir}/blaze/config"
param_filename=${blaze_dir}/blazemark/params/daxpy.prm
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.67.0-clang6.0.0/release/lib:$LD_LIBRARY_PATH
thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
vec_sizes_log=(2 3 4 5 6 7)
cache_sizes=(8 16 32 64 128 256)

rm -rf ${results_dir}/*.dat
benchmarks=('daxpy' 'dvecdvecadd')
r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h
size=$((10**vec_size))

rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp ${blaze_dir}/blaze/math/smp/hpx/DenseVector.h ${results_dir}/info
cp ${blaze_dir}/blaze/math/smp/hpx/Functions.h ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/
cp ${blaze_dir}/blazemark/scripts/hpx_configuration_test_counters_static.sh ${results_dir}/info/


export OMP_NUM_THREADS=1
for b in ${benchmarks[@]}
do
param_filename=${blaze_dir}/blazemark/params/$b.prm
for vec_size in ${vec_sizes_log[@]}
do
    size=$((10**vec_size))

for cache_size in ${cache_sizes[@]}
do
        git checkout $param_filename

	for p in $(seq 6)
	do
		if [ $p != $((vec_size-1)) ]
			then 
			        line_number=$((49+p))
				line="\"\\$(sed -n ${line_number}' p' $param_filename)"
				s='\/\/('
				sed -i "${line_number}s/(/${s}/" $param_filename

		fi
	done
        ./change_hpx_parameters.sh reset HPX.h
        ./change_hpx_parameters.sh CACHE_SIZE "${cache_size}"
	./generate_benchmarks.sh $b hpx
        
	threshold=$(sed -n '37 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
	echo "vector threshold:" ${threshold}
        cache_size_value=$(sed -n '118 p' "${cache_filename}")
        cache_size_value=$( echo ${cache_size_value:30:-4} )
	
        echo "cache size:" $cache_size_value
	for th in "${thr[@]}"
	do 
	     ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${b}-${th}-${r}-${cache_size}-${size}.dat
	    echo ${b} "benchmark for" ${r} "finished for "${th} "threads, chunk size ${c}, cache_size ${cache_size}"
	done
done
    echo "vector size ${size} finished"

done
    echo "${b} benchmark finished"
    
done
export LD_LIBRARY_PATH=${saved_path}


