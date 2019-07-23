#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
fi
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

rm -rf ${results_dir}/*.dat
benchmarks=('dmattdmatadd')
r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseMatrix.h

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
        elif [ $b == 'dmatdmatadd' ] || [ $b == 'dmattdmatadd' ]
        then
                start_line=91
                length=16
                end_line=119
        else
                echo "benchmark not specified"
        fi

	param_filename=${blaze_dir}/blazemark/params/$b.prm
	cd ${blaze_dir}
	git checkout $param_filename
        cd ${blazemark_dir}/scripts

	for line_number in $(seq 50 107)
		do			
			if [ $line_number -le $start_line ]
			then 
				s='\/\/('
		                sed -i "${line_number}s/(/${s}/" $param_filename
			else
                                sed -i "${line_number}s/)/,1)/" $param_filename
			fi
	done
	sed -i "58s/*/\//" $param_filename
	sed -i "${end_line}s/*/\//" $param_filename
	./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/"	               
	
	for th in "${thr[@]}"
		do 	
		    ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${node}-chunk_size-${b}-${th}.dat
		    echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
	done
        git checkout $param_filename
done
export LD_LIBRARY_PATH=${saved_path}

