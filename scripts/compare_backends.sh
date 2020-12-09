#!/bin/bash
backend="old"

if [ $# -eq 0 ]
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
	if [ $# -ge 2 ]
	then
		backend=$2
	fi
	if [ $# -eq 3 ] && [ "${backend}" == "new" ]
        then
                cs=$3
        fi
fi
echo "Running ${backend} backend on ${node}"

saved_path=$LD_LIBRARY_PATH
blaze_dir="/home/sshirzad/src/blaze_shahrzad"
repo_dir="/work/sshirzad"
blazemark_dir="${repo_dir}/repos/Blazemark"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}_main_apex/lib64"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main_apex/hpx_cmake_log.txt"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}_main_apex/info/"
results_dir="${blazemark_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blazemark_dir}/configurations"
#export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${hpx_dir}:"${hpx_dir}/hpx":$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8)
thr=(8)

benchmarks=('dmatdmatadd')
r='hpx'

#rm -rf ${results_dir}/*

export OMP_NUM_THREADS=1
for b in ${benchmarks[@]}
do
        #rm -rf ${results_dir}/info_${node}_${b}
        mkdir -p ${results_dir}/info_${node}_${b}/hpx_info
        cp -r ${hpx_log_dir}/* ${results_dir}/info_${node}_${b}/hpx_info
        hpx_source_dir="/home/sshirzad/src/hpx"
        cd $hpx_source_dir
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
        echo "$BRANCH branch">>${results_dir}/info_${node}_${b}/hpx_info/hpx_git.txt
        #
        git --git-dir $hpx_source_dir/.git log>>${results_dir}/info_${node}_${b}/hpx_info/hpx_git.txt

        cp ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info_${node}_${b}
        date>> ${results_dir}/info_${node}_${b}/date.txt
        #cp ${hpx_log_file} ${results_dir}/info/
        cp ${config_dir}/Configfile_hpx_${node}_install_main_apex ${results_dir}/info_${node}_${b}/
        cp ${blazemark_dir}/scripts/compare_backends.sh ${results_dir}/info_${node}_${b}/
        cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}_main/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}_${b}/
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

for line_number in $(seq 50 $((end_line-2)))
do
        s='\/\/('
        sed -i "${line_number}s/(/${s}/" $param_filename
                string=$(sed -n ${line_number}' p' $param_filename)
                if [[ $string != *",1"* ]]
                then
                        sed -i "${line_number}s/)/,1)/" $param_filename
                fi
done

line_number=$((end_line-1))
sed -i "58s/*/\//" $param_filename 
sed -i "${end_line}s/*/\//" $param_filename

sed -i "${line_number}s/*/\//" $param_filename
l=$(sed -n ${line_number}' p' $param_filename)

if [[ $l != *",1"* ]]
then
        mat_size=${l:1:-1}
else
        mat_size=${l:1:-3}
fi                                                                                     
mat_size=$(echo -e "${mat_size}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
echo "matrix size: $mat_size"

tag=""
#old backend
cd ${blazemark_dir}/scripts
if [ "$backend" == "old" ]
then 	
	./change_hpx_parameters.sh BLAZE_NEW_BACKEND 0
	tag=""
elif [ "$backend" == "new" ]
then
	./change_hpx_parameters.sh BLAZE_NEW_BACKEND 1
	./change_hpx_parameters.sh BLAZE_HPX_MATRIX_CHUNK_SIZE "${cs}"

	chunk_size=$(sed -n '49 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
	echo "chunk size:" ${chunk_size}
	
	block_size_value_row=$(sed -n '53 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
	block_size_value_col=$(sed -n '57 p' "${blaze_dir}/blaze/config/HPX.h"|cut -d' ' -f3)
	
	echo "block size row:" $block_size_value_row
	echo "block size col:" $block_size_value_col

	num_chunks_1=$(python3 -c "from math import ceil;print (int(ceil($mat_size/$block_size_value_row)))")
	num_chunks_2=$(python3 -c "from math import ceil;print (int(ceil($mat_size/$block_size_value_col)))")
	echo "num_chunks: "$((num_chunks_1*num_chunks_2))
	num_chunks=$((num_chunks_1*num_chunks_2))

	tag="_${block_size_value_row}_${block_size_value_col}_${chunk_size}"
else
	echo "wrong backend specified"
fi

$blazemark_dir/scripts/generate_benchmarks.sh ${b} ${r} "${blaze_dir}/blazemark/" ${node}
for th in ${thr[@]}
do
	mkdir -p ${results_dir}/otf2_${backend}/${th}
	cd ${results_dir}/otf2_${backend}/${th}
	
export APEX_OTF2=1
${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads/idle-rate  --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/cumulative-overhead --hpx:print-counter=/threads/count/cumulative --hpx:print-counter=/threads/time/average-overhead --hpx:print-counter='/papi{locality#*/worker-thread#*}/PAPI_L2_TCA' --hpx:print-counter='/papi{locality#*/worker-thread#*}/PAPI_L2_TCM'>>${results_dir}/${node}_${backend}-${b}-${th}-${r}-${mat_size}${tag}.dat
    
    echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
done
done    
export LD_LIBRARY_PATH=${saved_path}

