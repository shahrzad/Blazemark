#!/bin/bash
saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze"
repo_dir="/home/sshirzad/repos/Blazemark"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib64"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt"
results_dir="${repo_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${repo_dir}/configurations"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
#thr=(16)

rm -rf ${results_dir}/*
#benchmarks=('daxpy' 'dvecdvecadd')
benchmarks=('dvecdvecadd')
r='hpx'

rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/hpx_cmake_log.txt
cp $repo_dir/configurations/Configfile_hpx ${results_dir}/info/
cp ${repo_dir}/scripts/mat_hpx_ref.sh ${results_dir}/info/
git --git-dir $blaze_dir/.git log>>${results_dir}/info/blaze_git_log.txt
git --git-dir ~/src/hpx/.git log>>${results_dir}/info/hpx_git_log.txt

export OMP_NUM_THREADS=1
for b in ${benchmarks[@]}
do
if [ b == 'dvecdvecadd' ] || [ b == 'daxpy' ]
then 
end_line=219
elif [ $b == 'dmatdvecmult' ]
then
end_line=147
else
end_line=119
fi
param_filename=${blaze_dir}/blazemark/params/$b.prm
cd ${blaze_dir}

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
$repo_dir/scripts/generate_benchmarks.sh ${b} ${r} "${blaze_dir}/blazemark/"
for th in $(seq 16)
do
for i in $(seq 6)
do
#export HPX_COMMANDLINE_OPTIONS="--hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead"
    ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${i}-${b}-${th}-${r}.dat
    echo ${b} "benchmark for" ${r} "finished for "${th} "threads ${i}th time"
done
done    
done
export LD_LIBRARY_PATH=${saved_path}

