#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt"
results_dir="${blaze_dir}/blazemark/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blaze_dir}/blaze/config"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.67.0-clang6.0.0/release/lib:$LD_LIBRARY_PATH
thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
vec_sizes_log=(2 3 4 5 6 7)
chunk_sizes=(1 2 3 4 5 6 7 8 9)
multiplyers=(1)
cache_size=32
#chunk_sizes=(1 8 16 32 64 128 256 512)
#multiplyers=(1 2 4 8 10)

rm -rf ${results_dir}/*.dat
b='dvecdvecadd'
r='hpx'
param_filename=${blaze_dir}/blazemark/params/$b.prm

size=$((10**vec_size))

rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp ${blaze_dir}/blaze/math/smp/hpx/DenseVector.h ${results_dir}/info
cp ${blaze_dir}/blaze/math/smp/hpx/Functions.h ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/


export OMP_NUM_THREADS=1

for vec_size in ${vec_sizes_log[@]}
do
	size=$((10**vec_size))
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
	for cs in "${chunk_sizes[@]}"
	do
	for v in $(seq $vec_size)
	do
	#for m in "${multiplyers[@]}"
	#do
        c=$((cs*(10**(v-1))))
	if [ $c -le $((size/cache_size)) ]
	then
		./change_hpx_parameters.sh reset HPX.h
	        ./change_hpx_parameters.sh BLAZE_HPX_VECTOR_CHUNK_SIZE "${c}"
	#        ./change_hpx_parameters.sh BLAZE_HPX_VECTOR_THREADS_MULTIPLYER "${m}"
		./generate_benchmarks.sh $b hpx
		chunk_size=$(sed -n '41 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
		echo "chunk size:" ${chunk_size}
	
	#	multiplyer=$(sed -n '45 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
	#	echo "threads multiplyer:" ${multiplyer}
	
		threshold=$(sed -n '37 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
		echo "vector threshold:" ${threshold}
	
		for th in "${thr[@]}"
		do 
		    ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${b}-${th}-${r}-${chunk_size}-${size}.dat
		    echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
		done    
		#done
	fi
	done
done
done

#for vec_size in ${vec_sizes_log[@]}
#do
#        size=$((10**vec_size))
#
#        git checkout $param_filename
#
#	for p in $(seq 6)
#	do
#		if [ $p != $((vec_size-1)) ]
#			then 
#			        line_number=$((49+p))
#				line="\"\\$(sed -n ${line_number}' p' $param_filename)"
#				s='\/\/('
#				sed -i "${line_number}s/(/${s}/" $param_filename
#
#		fi
#	done
#
#	c=$((10**vec_size))
#	m=1
#	./change_hpx_parameters.sh reset HPX.h
#	./change_hpx_parameters.sh BLAZE_HPX_VECTOR_CHUNK_SIZE "${c}"
#	#./change_hpx_parameters.sh BLAZE_HPX_VECTOR_THREADS_MULTIPLYER "${m}"
#	./generate_benchmarks.sh $b hpx
#	chunk_size=$(sed -n '41 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
#	echo "chunk size:" ${chunk_size}
#	
#	#multiplyer=$(sed -n '45 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
#	#echo "threads multiplyer:" ${multiplyer}
#	
#	threshold=$(sed -n '37 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
#	echo "vector threshold:" ${threshold}
#	
#	for th in "${thr[@]}"
#	        do
#	            ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${b}-${th}-${r}-${chunk_size}-${size}.dat
#	            echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
#        done
#done

export LD_LIBRARY_PATH=${saved_path}

