#!/bin/bash

saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
results_dir="/home/sshirzad/Blazemark/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blaze_dir}/blaze/config"
thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
vec_sizes_log=(2 3 4 5 6 7)

rm -rf ${results_dir}/*

benchmarks=('daxpy' 'dvecdvecadd')
r='openmp'
for b in ${benchmarks[@]}
do
param_filename=${blaze_dir}/blazemark/params/$b.prm

git checkout $param_filename

for p in $(seq 6)
do
	line_number=$((49+p))
	s='\/\/('
	sed -i "${line_number}s/(/${s}/" $param_filename
done
sed -i "58s/*/\//" $param_filename 
sed -i "219s/*/\//" $param_filename

./generate_benchmarks.sh ${b} ${r}
for th in ${thr[@]}
do
for i in $(seq 11)
do
export OMP_NUM_THREADS=${th} 
${benchmarks_dir}/${b}_${r} -only-blaze>>${results_dir}/${i}-${b}-${th}-${r}.dat
echo "${i}th time finished for ${th} threads"
done
done
done
