#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib64"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt"
results_dir="${blazemark_dir}/results/dmatdvecadd"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blazemark_dir}/configurations"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
thr=(1 2 4 8 10 12 14 16)
#thr=(16)
#thr=(1 2 4 7 8 7 16)
#vec_sizes_log=(2 3 4 5 6 7)
chunk_sizes=(1 5 10 15 20)
#block_sizes=(4 8 16 32)
block_sizes_row=(32 64 128 256 512 1024 2048)

#rm -rf ${results_dir}/*.dat
#benchmarks=('daxpy' 'dvecdvecadd')
benchmarks=('dmatdvecmult')
r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h

mkdir -p ${results_dir}
#rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/
cp ${config_dir}/Configfile_hpx ${results_dir}/info/
cp ${blazemark_dir}/scripts/mat_hpx_vecmat.sh ${results_dir}/info/
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

#        git checkout $param_filename
#
#	for p in $(seq 6)
#	do
#		line_number=$((49+p))
#		s='\/\/('
#		sed -i "${line_number}s/(/${s}/" $param_filename
#	done
#        sed -i "58s/*/\//" $param_filename 
#        sed -i "${end_line}s/*/\//" $param_filename

	for c in "${chunk_sizes[@]}"
	do
		#./change_hpx_parameters.sh reset HPX.h
	        ./change_hpx_parameters.sh BLAZE_HPX_VECTOR_CHUNK_SIZE "${c}"
                ./change_hpx_parameters.sh BLAZE_HPX_VECTOR_BLOCK_SIZE "${block_size_row}"
		./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/"
               
		chunk_size=$(sed -n '41 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
		echo "chunk size:" ${chunk_size}
	
                block_size_value_row=$(sed -n '45 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)

                echo "block size:" $block_size_value_row

		for th in "${thr[@]}"
		do 

		    ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive >>${results_dir}/${i}-${b}-${th}-${r}-${chunk_size}-${block_size_row}.dat
		    echo ${b} "benchmark for" ${r} "finished for "${th} "threads, chunk size ${c}, block_size: ${block_size_row}"
		done
		done    
	done
done
export LD_LIBRARY_PATH=${saved_path}

