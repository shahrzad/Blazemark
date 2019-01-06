#!/bin/bash

saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
repo_dir="/home/sshirzad/repos/Blazemark"
results_dir="$repo_dir/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blaze_dir}/blaze/config"
#thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
thr=(16)
vec_sizes_log=(2 3 4 5 6 7)

#rm -rf ${results_dir}/*

#benchmarks=('daxpy' 'dvecdvecadd')
benchmarks=('dmatdmatadd')
r='openmp'
for b in ${benchmarks[@]}
do
#if [ b == 'dvecdvecadd' ] || [ b == 'daxpy' ]
#then 
#end_line=219
#else
#end_line=119
#fi
#param_filename=${blaze_dir}/blazemark/params/$b.prm
#echo $param_filename
#cd ${blaze_dir}
#git checkout $param_filename
#
#for p in $(seq 6)
#do
#	line_number=$((49+p))
#	s='\/\/('
#	sed -i "${line_number}s/(/${s}/" $param_filename
#done
#sed -i "58s/*/\//" $param_filename 
#sed -i "${end_line}s/*/\//" $param_filename
#
$repo_dir/scripts/generate_benchmarks.sh ${b} ${r}
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
