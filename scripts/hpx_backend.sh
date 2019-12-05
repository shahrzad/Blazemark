#!/bin/bash
#blaze_id branch benchmark node
steps=0
if [ $# -ge 1 ]
then
	blaze_id=$1
	if [ $# -ge 2 ]
	then
		branch=$2
		if [ $# -ge 3 ]
		then
			benchmark=$3
               		if [ $# -ge 4 ]
	                then
				node=$4
			else
				node="marvin"
			fi
		else
			benchmark="all"
			node="marvin"
		fi
	else
	        branch="hpx_backend"
		benchmark="all"
        	node="marvin"
	fi
else
        blaze_id="blaze_shahrzad"
        benchmark="all"
        node="marvin"
fi

if [ $benchmark == "all" ]
then
        benchmarks=('dmatdmatmult' 'dmattdmatmult' 'dmatdmatadd' 'dmattdmatadd' 'dmatdvecmult' )
else
        benchmarks=( $benchmark )
fi

echo -e "blaze_id:$blaze_id \nbranch:$branch \nnode:$node \nbenchmarks:$benchmark"

saved_path=$LD_LIBRARY_PATH
blazemark_dir="/home/sshirzad/repos/Blazemark"
blaze_dir="/home/sshirzad/src/$blaze_id"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_$node/lib64"
hpx_log_file="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_$node/info/*"
results_dir="${blazemark_dir}/results"
benchmarks_dir="${blaze_dir}/blazemark/benchmarks"
config_dir="${blazemark_dir}/configurations"
hpx_source_dir="/home/sshirzad/src/hpx"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.68.0-clang6.0.1/release/lib:$LD_LIBRARY_PATH
#thr=(1 4 8 16)
thr=(1 2 3 4 5 6 7 8)

r='hpx'
cache_filename=${blaze_dir}/blaze/math/smp/hpx/DenseMatrix.h

rm -rf ${results_dir}/*.dat
rm -rf ${results_dir}/info
mkdir ${results_dir}/info

cp -r ${blaze_dir}/blaze/math/smp/hpx/* ${results_dir}/info
date>> ${results_dir}/info/date.txt
cp ${hpx_log_file} ${results_dir}/info/
cp ${config_dir}/Configfile_hpx ${results_dir}/info/
cp ${blazemark_dir}/scripts/hpx_backend.sh ${results_dir}/info/
cp /home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp_$node/include/hpx/parallel/util/detail/chunk_size.hpp ${results_dir}/info/
git --git-dir $hpx_source_dir/.git log>>${results_dir}/info/hpx_git.txt
cd ${blaze_dir}
BR=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BR" != $branch ]]; then
        git checkout $branch
fi
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "$BR branch">>${results_dir}/info/blaze_git.txt
git --git-dir $blaze_dir/.git log>>${results_dir}/info/blaze_git.txt

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
	elif [ $b == 'dmatdmatadd' ] || [ $b == 'dmattdmatadd' ] || [ $b == 'dmatdmatdmatadd' ]
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

	for line_number in $(seq 50 $((end_line-1)))
		do			
		if [ $line_number -le $start_line ] || [ $line_number -gt $((start_line+length)) ]
		then
			s='\/\/('
		        sed -i "${line_number}s/(/${s}/" $param_filename			
		else
			if [ $steps == 1 ]
                        then
				string=$(sed -n ${line_number}' p' $param_filename)
				if [[ $string != *",1"* ]]
				then
                                	sed -i "${line_number}s/)/,1)/" $param_filename
				fi
                        fi
		fi
	done
	sed -i "58s/*/\//" $param_filename
	sed -i "${end_line}s/*/\//" $param_filename
	./generate_benchmarks.sh $b hpx "${blaze_dir}/blazemark/" $node	               
	
	for th in "${thr[@]}"
		do 	
		    ${benchmarks_dir}/${b}_${r}_${node} -only-blaze --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive>>${results_dir}/${node}-${b}-${th}-${branch}.dat
		    echo ${b} "benchmark for" ${r} "finished for "${th} "threads on $branch branch"
	done
        cd ${blaze_dir}
	git checkout $param_filename
done
export LD_LIBRARY_PATH=${saved_path}

