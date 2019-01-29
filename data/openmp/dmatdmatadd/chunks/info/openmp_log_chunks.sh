#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
blazemark_dir="/home/sshirzad/repos/Blazemark"
results_dir="$blazemark_dir/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blaze_dir}/blaze/config"
#thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
thr=(16)
chunk_sizes=(10)
block_sizes_row=(8)
block_sizes_col=(512)
rm -rf ${results_dir}/*

#benchmarks=('daxpy')
#('dvecdvecadd')
benchmarks=('dmatdmatadd')
r='openmp'
for b in ${benchmarks[@]}
do
if [ $b == 'dvecdvecadd' ] || [ $b == 'daxpy' ]
then 
end_line=219
else
end_line=119
fi
param_filename=${blaze_dir}/blazemark/params/$b.prm
echo $param_filename
cd ${blaze_dir}
git checkout $param_filename

for p in $(seq 6)
do
	line_number=$((49+p))
	s='\/\/('
	sed -i "${line_number}s/(/${s}/" $param_filename
done
sed -i "58s/*/\//" $param_filename 
sed -i "${end_line}s/*/\//" $param_filename
cd ${blazemark_dir}/scripts
mkdir -p ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp $blazemark_dir/scripts/openmp_log_chunks.sh ${results_dir}/info/
cp $blaze_dir/blaze/math/smp/openmp/* ${results_dir}/info/

for block_size_row in ${block_sizes_row[@]}
do
for block_size_col in ${block_sizes_col[@]}
do
	for c in "${chunk_sizes[@]}"
	do
echo ${c}
echo ${block_size_row}
echo ${block_size_col}
pwd
	        ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_CHUNK_SIZE "${c}"
                ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW "${block_size_row}"
                ./change_hpx_parameters.sh BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN "${block_size_col}"
		./generate_benchmarks.sh $b hpx
               
		chunk_size=$(sed -n '49 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
		echo "chunk size:" ${chunk_size}
	
                block_size_value_row=$(sed -n '53 p' "${config_dir}/HPX.h"|cut -d' ' -f3)
                block_size_value_col=$(sed -n '57 p' "${config_dir}/HPX.h"|cut -d' ' -f3)

                echo "block size row:" $block_size_value_row
                echo "block size col:" $block_size_value_col

for th in ${thr[@]}
do
for i in $(seq 1)
do
export OMP_NUM_THREADS=${th} 
export OMP_PLACES=cores
${benchmarks_dir}/${b}_${r} -only-blaze>>${results_dir}/${i}-${b}-${th}-${r}-${chunk_size}-${block_size_row}-${block_size_col}.dat

echo "${i}th time finished for ${th} threads"
done
done
done
done
done
done

