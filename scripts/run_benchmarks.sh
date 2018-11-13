#!/bin/bash
saved_path=$LD_LIBRARY_PATH
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_no_hpxmp/lib"
results_dir="/home/sshirzad/src/blaze_shahrzad/blazemark/results"
benchmarks_dir="/home/sshirzad/src/blaze_shahrzad/blazemark/benchmarks"
export LD_LIBRARY_PATH=${hpx_dir}:/opt/boost/1.67.0-clang6.0.0/release/lib:$LD_LIBRARY_PATH
thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

rm -rf ${results_dir}/*.dat
if [ $# -eq 0 ] 
then 
    benchmarks=('daxpy' 'dvecdvecadd' 'dmatsvecmult' 'dmattdmatadd' 'dmattdmatmult' 'dmatsmatmult' 'smatdmatmult' 'dmattrans')
    runtimes=('hpx' 'openmp' 'cpp' 'boost')
elif [ $# -eq 1 ]
then
   benchmarks=$1
   runtimes=('hpx' 'openmp' 'cpp' 'boost')
elif [ $# -eq 2 ]
then 
    benchmarks=$1
    runtimes=$2
fi


for b in "${benchmarks[@]}"
do
for r in "${runtimes[@]}"
do
for th in "${thr[@]}"
do 
    if [ ${r} == 'hpx' ]
        then
           chunk_size=$(sed -n '41 p' /home/sshirzad/src/blaze_shahrzad/blaze/blaze/config/HPX.h|cut -d' ' -f3)
           echo "chunk size:" ${chunk_size}

           multiplyer=$(sed -n '45 p' /home/sshirzad/src/blaze_shahrzad/blaze/blaze/config/HPX.h|cut -d' ' -f3)
           echo "threads multiplyer:" ${multiplyer}

           threshold=$(sed -n '37 p' /home/sshirzad/src/blaze_shahrzad/blaze/blaze/config/HPX.h|cut -d' ' -f3)
           echo "vector threshold:" ${threshold}

           export OMP_NUM_THREADS=1
           ${benchmarks_dir}/${b}_${r} -only-blaze --hpx:threads=${th}>>${results_dir}/${b}-${th}-${r}-${chunk_size}-${multiplyer}-${threshold}.dat
    else 
        if [ ${r} == 'openmp' ]
            then
            export OMP_NUM_THREADS=${th}
        elif [ ${r} == 'cpp' ] || [ ${r} == 'boost' ]
            then
            export BLAZE_NUM_THREADS=${th} 
        fi
        ${benchmarks_dir}/${b}_${r} -only-blaze>>${results_dir}/${b}-${th}-${r}.dat
    fi
    echo ${b} "benchmark for" ${r} "finished for "${th} "threads"
done    
done
done

export LD_LIBRARY_PATH=${saved_path}

