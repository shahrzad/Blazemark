#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
echo "Running on ${node}"
fi
papi=1
steps=0
counters=0
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
#hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib64"
hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/lib"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/info/"
results_dir="${blazemark_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blazemark_dir}/configurations"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
thr=(1 2 3 4 5 6 7 8)
chunk_sizes=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 1200 1380 1587 1800 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)
block_sizes_row=(6)
block_sizes_col=(128 256 512 1024)
rm -rf ${results_dir}/*.dat
benchmarks=('dmatdmatadd')
r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseMatrix.h

i=1

export OMP_NUM_THREADS=1
for b in ${benchmarks[@]}
	do
	rm -rf ${results_dir}/info_${node}_${b}
	mkdir -p ${results_dir}/info_${node}_${b}/hpx_info
	cp -r $hpx_log_dir/* ${results_dir}/info_${node}_${b}/hpx_info
	#hpx_source_dir="/home/sshirzad/src/hpx"
	#cd $hpx_source_dir
	#BRANCH=$(git rev-parse --abbrev-ref HEAD)
	#echo "$BRANCH branch">>${results_dir}/info/hpx_git.txt
	#
	#git --git-dir $hpx_source_dir/.git log>>${results_dir}/info/hpx_git.txt
	
	cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info_${node}_${b}
	date>> ${results_dir}/info_${node}_${b}/date.txt
	#cp ${hpx_log_file} ${results_dir}/info/
	cp ${config_dir}/Configfile_hpx_${node} ${results_dir}/info_${node}_${b}/
	cp ${blazemark_dir}/scripts/mat_hpx_chunks.sh ${results_dir}/info_${node}_${b}/
	cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}_${b}/
	git --git-dir $blaze_dir/.git log>>${results_dir}/info_${node}_${b}/blaze_git.txt
	cd ${blaze_dir}
	BRANCH=$(git rev-parse --abbrev-ref HEAD)
	#if [[ "$BRANCH" != "master" ]]; then
	#        git checkout master
	#fi
	#BRANCH=$(git rev-parse --abbrev-ref HEAD)
	echo "$BRANCH branch">>${results_dir}/info_${node}_${b}/blaze_git.txt
	git --git-dir $blaze_dir/.git log>>${results_dir}/info_${node}_${b}/blaze_git.txt

	if [ $b == 'dvecdvecadd' ] || [ $b == 'daxpy' ]
	then 
		end_line=219
	elif [ $b == 'dmatdmatmult' ] || [ $b == 'dmattdmatmult' ]
	then 
		start_line=81
		length=37
                end_line=119
	elif [ $b == 'dmatdvecmult' ]
	then
		start_line=111
		length=35
		end_line=147
	elif [ $b == 'dmatdmatadd' ] || [ $b == 'dmattdmatadd' ] || [ $b == 'dmatdmatdmatadd' ]
        then
                start_line=91
                #length=16
		#length=10
		length=2
                end_line=119
		start_line=100
		#start_line=117
		
        else
		echo "benchmark not specified"
	fi
	param_filename=${blaze_dir}/blazemark/params/$b.prm

	cd ${blaze_dir}
	git checkout $param_filename
        for line_number in $(seq 50 $((end_line-1)))
                do
                #if [ $line_number -le $start_line ] || [ $line_number -gt $((start_line+length)) ]
                #then
                s='\/\/('
                sed -i "${line_number}s/(/${s}/" $param_filename
                #else
                if [ $steps == 1 ]
                then
                        string=$(sed -n ${line_number}' p' $param_filename)
                        if [[ $string != *",1"* ]]
                        then
                                sed -i "${line_number}s/)/,1)/" $param_filename
                        fi
                 fi
                #fi
        done
        sed -i "58s/*/\//" $param_filename
        sed -i "${end_line}s/*/\//" $param_filename

	for block_size_row in ${block_sizes_row[@]}
	do
	        cd ${blazemark_dir}/scripts
		./change_hpx_parameters.sh BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW "${block_size_row}"
	
		for block_size_col in ${block_sizes_col[@]}
		do
		        cd ${blazemark_dir}/scripts
			./change_hpx_parameters.sh BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN "${block_size_col}"

			for line_number in $(seq $((start_line+1)) $((start_line+length)))
		        do
		                s='\/\/('
		                sed -i "${line_number}s/${s}/(/" $param_filename
		                l=$(sed -n ${line_number}' p' $param_filename)
		                if [[ $l != *",1"* ]]
		                then
		                        mat_size=${l:1:-1}
		                else
		                        mat_size=${l:1:-3}
		                fi                                                                                                                  
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
						        cd ${blazemark_dir}/scripts
						        ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_CHUNK_SIZE "${c}"
				                        ./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/" ${node}
							chunk_size=$(sed -n '49 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
							echo "chunk size:" ${chunk_size}	
		
					                block_size_value_row=$(sed -n '53 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
					                block_size_value_col=$(sed -n '57 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
	
					                echo "block size row:" $block_size_value_row
					                echo "block size col:" $block_size_value_col	
	
							for th in "${thr[@]}"
								do 	
									if [ $counters == 1 ]
									then
									       ${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead --hpx:print-counter='/papi{locality#*/worker-thread#*}/PAPI_L2_TCA' --hpx:print-counter='/papi{locality#*/worker-thread#*}/PAPI_L2_TCM'>>${results_dir}/${node}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}-${mat_size}.dat
									else
										${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive>>${results_dir}/${node}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}-${mat_size}.dat
									fi
								    echo ${b} "benchmark for" ${r} "finished for "${th} "threads, chunk size ${c}, block_size row: ${block_size_row} col:${block_size_col} matrix size: $mat_size"
							done
					      fi
				done
				c=$num_chunks
			        cd ${blazemark_dir}/scripts
		                ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_CHUNK_SIZE "${c}"
                       		./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/" ${node}
				chunk_size=$(sed -n '49 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
		                echo "chunk size:" ${chunk_size}	
		                block_size_value_row=$(sed -n '53 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
		                block_size_value_col=$(sed -n '57 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)	
	        	        echo "block size row:" $block_size_value_row
		                echo "block size col:" $block_size_value_col
	
		                for th in "${thr[@]}"
	        	        do
					if [ $counters == 1 ]
                                        then
                                               ${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead --hpx:print-counter='/papi{locality#*/worker-thread#*}/PAPI_L2_TCA' --hpx:print-counter='/papi{locality#*/worker-thread#*}/PAPI_L2_TCM'>>${results_dir}/${node}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}-${mat_size}.dat
                                        else
                                               ${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive>>${results_dir}/${node}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}-${mat_size}.dat
                                        fi

					echo ${b} "benchmark for" ${r} "finished for "${th} "threads, chunk size ${c}, block_size row: ${block_size_row} col:${block_size_col} matrix size: ${mat_size}"
				done	    
		                sed -i "${line_number}s/(/${s}/" $param_filename
			done
		done
	done
done
export LD_LIBRARY_PATH=${saved_path}

