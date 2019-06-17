#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt"
results_dir="${blazemark_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blaze_dir}/blaze/config"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.67.0-clang6.0.0/release/lib:$LD_LIBRARY_PATH
#thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
thr=(16)

rm -rf ${results_dir}/*.dat
#benchmarks=('daxpy' 'dvecdvecadd')
benchmarks=('dmatdmatadd')
r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h

rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/
cp ${blaze_dir}/blazemark/configurations/Configfile_hpx ${results_dir}/info/
cp ${blazemark_dir}/scripts/mat_hpx_configuration_test_counters_cachesize_log_ref.sh ${results_dir}/info/
git --git-dir $blaze_dir/.git log>>${results_dir}/info/

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

./generate_benchmarks.sh $b hpx

for th in "${thr[@]}"
do 
for i in $(seq 1)
do

    ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${i}-${b}-${th}-${r}.dat
    echo ${b} "benchmark for" ${r} "finished for "${th} "threads ${i}th time"
done
done    
done
export LD_LIBRARY_PATH=${saved_path}

