#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib64"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt"
results_dir="${blazemark_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blazemark_dir}/configurations"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
#thr=(1 4 8 16)
thr=(1 4 8 16)
#thr=(1 2 4 7 8 7 16)
#vec_sizes_log=(2 3 4 5 6 7)
chunk_sizes=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 1200 1380 1587)
#chunk_sizes=(2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 25000)

#block_sizes=(4 8 16 32)
block_sizes_row=(64)
block_sizes_col=(64)

rm -rf ${results_dir}/*.dat
benchmarks=('dmatdmatadd')
r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h

rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/
cp ${config_dir}/Configfile_hpx ${results_dir}/info/
cp ${blazemark_dir}/scripts/mat_hpx.sh ${results_dir}/info/
cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info/
git --git-dir $blaze_dir/.git log>>${results_dir}/info/hpx_git.txt
i=1

export OMP_NUM_THREADS=1
for b in ${benchmarks[@]}
	do
	if [ b == 'dvecdvecadd' ] || [ b == 'daxpy' ]
	then 
		end_line=219
	else
		end_line=119
	fi
	param_filename=${blaze_dir}/blazemark/params/$b.prm
	
	
	for block_size_row in ${block_sizes_row[@]}
	do
		./change_hpx_parameters.sh BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW "${block_size_row}"
	
		for block_size_col in ${block_sizes_col[@]}
		do
			./change_hpx_parameters.sh BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN "${block_size_col}"
	
			for p in $(seq 16)
		    	do
		                cd ${blaze_dir}
	                        git checkout $param_filename
        	                cd ${blazemark_dir}/scripts

	        		for q in $(seq 70)
	        			do
	               				line_number=$((49+q))
						s='\/\/('
				                sed -i "${line_number}s/(/${s}/" $param_filename
	        		done
			        sed -i "58s/*/\//" $param_filename
	        		sed -i "${end_line}s/*/\//" $param_filename
	
		        	line_number=$((91+p))	        
		        	s='\/\/('
		        	sed -i "${line_number}s/${s}/(/" $param_filename
		        	l=$(sed -n "${line_number} p" "$param_filename")
		        	mat_size=${l:1:-1}
	        		mat_size=$(echo -e "${mat_size}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
	        		num_chunks_1=$(python -c "from math import ceil;print (ceil($mat_size/$block_size_row))")
	        		num_chunks_2=$(python -c "from math import ceil;print (ceil($mat_size/$block_size_col))")
		        	num_chunks=$((num_chunks_1*num_chunks_2))
		        	echo "matrix size: $mat_size num_chunks: "$((num_chunks_1*num_chunks_2))
				for c in "${chunk_sizes[@]}"
					do
						if [ $c -lt $num_chunks ]
						then
		
							#./change_hpx_parameters.sh reset HPX.h
						        ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_CHUNK_SIZE "${c}"
							./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/"	               
							chunk_size=$(sed -n '49 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
							echo "chunk size:" ${chunk_size}	
		
					                block_size_value_row=$(sed -n '53 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
					                block_size_value_col=$(sed -n '57 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
	
					                echo "block size row:" $block_size_value_row
					                echo "block size col:" $block_size_value_col	
	
							for th in "${thr[@]}"
								do 	
								    ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${i}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}-${mat_size}.dat
								    echo ${b} "benchmark for" ${r} "finished for "${th} "threads, chunk size ${c}, block_size row: ${block_size_row} col:${block_size_col} matrix size: $mat_size"
							done
					      fi
				done
				c=$num_chunks
		                ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_CHUNK_SIZE "${c}"
				./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/"	
	        	        chunk_size=$(sed -n '49 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
		                echo "chunk size:" ${chunk_size}	
		                block_size_value_row=$(sed -n '53 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
		                block_size_value_col=$(sed -n '57 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)	
	        	        echo "block size row:" $block_size_value_row
		                echo "block size col:" $block_size_value_col
	
		                for th in "${thr[@]}"
	        	        do
					${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${i}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}-${mat_size}.dat
					echo ${b} "benchmark for" ${r} "finished for "${th} "threads, chunk size ${c}, block_size row: ${block_size_row} col:${block_size_col} matrix size: ${mat_size}"
				done	    
			done
		done
	done
done
export LD_LIBRARY_PATH=${saved_path}

