#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
ps=$2
else
node=$1
echo "Running on ${node}"
fi
saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
hpx_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/lib"
hpx_bin_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/bin"
hpx_source_dir="/home/sshirzad/src/hpx"
hpx_log_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_${node}/info/"
results_dir="${blazemark_dir}/results_grain_size"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH

thr=(1 2 3 4 5 6 7 8)
problem_sizes=(250000)

#mkdir -p ${results_dir}
#rm -rf ${results_dir}/*.dat
#rm -rf ${results_dir}/info_${node}/
mkdir -p ${results_dir}/info_${node}/hpx_info
date>> ${results_dir}/info_${node}/date.txt
cp ${blazemark_dir}/scripts/grain_size_test.sh ${results_dir}/info_${node}/
cp -r $hpx_log_dir/* ${results_dir}/info_$node/hpx_info
cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_${node}/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info_${node}/hpx_info/

#for ps in ${problem_sizes[@]}
for ps in $(seq 10000000 1000000 20000000)
do
        echo "problem size ${ps}"
	for il in $(seq ${ps})
                do
                if [ $((ps%il)) -eq 0 ]
                then
                        echo " task length ${il}"
                        ni=$((ps/il))

		echo " number of iterationse ${ni}"
#	for th in ${thr[@]}
#		do
#		echo "${th} threads"

		export OMP_NUM_THREADS=1

		${hpx_bin_dir}/grain_size_test  -Ihpx.stacks.use_guard_pages=0 --num_iterations=${ni}  --iter_length=${il} --repetitions=6 --seq>>${results_dir}/${node}_seq_grain_size_${il}_${ni}.dat
		echo "Run sequentially for ${ni} iterations, iter length of ${il}"			
	#done
		fi
	done
done

