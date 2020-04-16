#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
if [ $node = "marvin_old" ]
then
        module load clang/6.0.1
        module load boost/1.68.0-clang6-release
fi

echo "Running on ${node}"
fi

blazemark_dir="/home/sshirzad/repos/Blazemark"
saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
repo_dir="/home/sshirzad/repos/Blazemark"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}/lib64"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/hpx_cmake_log.txt"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/info/"
results_dir="${repo_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${repo_dir}/configurations"
#export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${hpx_dir}:$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8)
#thr=(16)

#rm -rf ${results_dir}/*
#benchmarks=('daxpy' 'dvecdvecadd')
benchmarks=('dmatdmatdmatadd')
r='hpx'

#rm -rf ${results_dir}/info
#mkdir ${results_dir}/info
#
#cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info
#date>> ${results_dir}/info/date.txt
#cp ${hpx_log_file} ${results_dir}/info/hpx_cmake_log.txt
#cp $repo_dir/configurations/Configfile_hpx ${results_dir}/info/
#cp ${repo_dir}/scripts/mat_hpx_ref.sh ${results_dir}/info/
#git --git-dir $blaze_dir/.git log>>${results_dir}/info/blaze_git_log.txt
#git --git-dir ~/src/hpx/.git log>>${results_dir}/info/hpx_git_log.txt
#
export OMP_NUM_THREADS=1
for b in ${benchmarks[@]}
do
        rm -rf ${results_dir}/info_${node}_${b}
        mkdir -p ${results_dir}/info_${node}_${b}/hpx_info
        cp -r ${hpx_log_dir}/* ${results_dir}/info_${node}_${b}/hpx_info
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
        cp ${blazemark_dir}/scripts/mat_hpx_ref.sh ${results_dir}/info_${node}_${b}/
        cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}_${b}/
        git --git-dir $blaze_dir/.git log>>${results_dir}/info_${node}_${b}/blaze_git.txt
        cd ${blaze_dir}
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
	
if [ b == 'dvecdvecadd' ] || [ b == 'daxpy' ]
then 
end_line=219
elif [ $b == 'dmatdvecmult' ]
then
end_line=147
else
start_line=92
end_line=119
fi
param_filename=${blaze_dir}/blazemark/params/$b.prm
cd ${blaze_dir}

        git checkout $param_filename

#	for p in $(seq 6)
#	do
#		line_number=$((49+p))
#		s='\/\/('
#		sed -i "${line_number}s/(/${s}/" $param_filename
#	done

	for line_number in $(seq 50 $((start_line-1)))
        do
                s='\/\/('
                sed -i "${line_number}s/(/${s}/" $param_filename
        done
        sed -i "58s/*/\//" $param_filename 
        sed -i "${end_line}s/*/\//" $param_filename

$blazemark_dir/scripts/generate_benchmarks.sh ${b} ${r} "${blaze_dir}/blazemark/" ${node}
for th in $(seq 8)
do
#export HPX_COMMANDLINE_OPTIONS="--hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead"
${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive>>${results_dir}/${node}-${b}-${th}-${r}.dat
    
#${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th} --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead>>${results_dir}/${i}-${b}-${th}-${r}.dat
    echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
done
done    
export LD_LIBRARY_PATH=${saved_path}

