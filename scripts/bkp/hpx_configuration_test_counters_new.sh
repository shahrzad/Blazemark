#!/bin/bash
saved_path=$LD_LIBRARY_PATH
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib"
results_dir="/home/sshirzad/src/blaze_pdiehl/blaze/blazemark/results"
benchmarks_dir="/home/sshirzad/src/blaze_pdiehl/blaze/blazemark/benchmarks"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.67.0-clang6.0.0/release/lib:$LD_LIBRARY_PATH
thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
vec_size=1000000
num_points=50
sec_size=$((vec_size/num_points))

multiplyers=(1)
#chunk_sizes=(1 8 16 32 64 128 256 512)
#multiplyers=(1 2 4 8 10)

rm -rf ${results_dir}/*.dat
b='daxpy'
r='hpx'

export OMP_NUM_THREADS=1
for m in "${multiplyers[@]}"
do
        ./change_hpx_parameters.sh reset
	./change_hpx_parameters.sh BLAZE_HPX_VECTOR_THREADS_MULTIPLYER "${m}"

	multiplyer=$(sed -n '45 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
        echo "threads multiplyer:" ${multiplyer}

        threshold=$(sed -n '37 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
        echo "vector threshold:" ${threshold}

	for th in "${thr[@]}"
        do
		for ((k=0; k<num_points; k++));
		do
	        	if [ $k == 0 ]
			then 
				c=1
			else
				c=$((k*vec_size/(th*num_points)))
			fi
		       	./change_hpx_parameters.sh BLAZE_HPX_VECTOR_CHUNK_SIZE "${c}"
		        ./generate_benchmarks.sh daxpy hpx
        		chunk_size=$(sed -n '41 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
	       		echo "chunk size:" ${chunk_size}

	        	multiplyer=$(sed -n '45 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
		       	echo "threads multiplyer:" ${multiplyer}

		       	threshold=$(sed -n '37 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
		        echo -e "vector threshold: ${threshold} \n"
	
	        	${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${b}-${th}-${r}-${chunk_size}-${multiplyer}-${threshold}.dat
			
                        if [ $th != 1 ]
		        then
		        	c=$((vec_size/th+k*2*vec_size*(th-1)/(th*num_points)))
                                ./change_hpx_parameters.sh BLAZE_HPX_VECTOR_CHUNK_SIZE "${c}"
                                ./generate_benchmarks.sh daxpy hpx
                                chunk_size=$(sed -n '41 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
                                echo "chunk size:" ${chunk_size}
                                
                                multiplyer=$(sed -n '45 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
                                echo "threads multiplyer:" ${multiplyer}
                                
                                threshold=$(sed -n '37 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
                                echo -e "vector threshold:" ${threshold} "\n"
                                
                                ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${b}-${th}-${r}-${chunk_size}-${multiplyer}-${threshold}.dat
                       fi 
		done
        	echo -e ${b} "benchmark for" ${r} "finished for "${th} "threads\n"
	done
done



c=$vec_size
m=1
./change_hpx_parameters.sh reset
./change_hpx_parameters.sh BLAZE_HPX_VECTOR_CHUNK_SIZE "${c}"
./change_hpx_parameters.sh BLAZE_HPX_VECTOR_THREADS_MULTIPLYER "${m}"
./generate_benchmarks.sh daxpy hpx
chunk_size=$(sed -n '41 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
echo "chunk size:" ${chunk_size}

multiplyer=$(sed -n '45 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
echo "threads multiplyer:" ${multiplyer}

threshold=$(sed -n '37 p' /home/sshirzad/src/blaze_pdiehl/blaze/blaze/config/HPX.h|cut -d' ' -f3)
echo "vector threshold:" ${threshold}

for th in "${thr[@]}"
        do
            ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${b}-${th}-${r}-${chunk_size}-${multiplyer}-${threshold}.dat
            echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
        done

export LD_LIBRARY_PATH=${saved_path}






