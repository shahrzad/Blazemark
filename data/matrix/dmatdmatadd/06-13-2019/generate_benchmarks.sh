#!/bin/bash

#blazemark_dir="/home/sshirzad/src/blaze_shahrzad/blazemark"
blazemark_dir="/home/sshirzad/src/blaze/blazemark"
config_dir="/home/sshirzad/repos/Blazemark/configurations"

if [ $# -eq 3 ]
then
    benchmarks=$1
    runtimes=$2
    blazemark_dir=$3
#    if [ ${runtimes} == 'hpx' ]
#    then
##        rm -rf ${results_dir}/hpx
##        mkdir ${results_dir}/hpx
#        #date>>${results_dir}/hpx/date.txt
#        #git --git-dir ~/src/hpx/.git log>>${results_dir}/hpx/hpx_git_log.txt
#
##        cp ~/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt ${results_dir}/hpx/hpx_cmake_log.txt
#    fi
else
    if [ $# -eq 0 ] 
    then 
        #benchmarks=('daxpy' 'dvecdvecadd' 'dmatsvecmult' 'dmattdmatadd' 'dmattdmatmult' 'dmatsmatmult' 'smatdmatmult' 'dmattrans')
         benchmarks=('daxpy' 'dvecdvecadd')
    fi
    if [ $# -eq 1 ]
    then
        benchmarks=$1
    fi
    runtimes=('hpx' 'openmp' 'cpp' 'boost')
#    rm -rf ${results_dir}/hpx
#    mkdir ${results_dir}/hpx
    #date>>${results_dir}/hpx/date.txt
    #git --git-dir ~/src/hpx/.git log>>${results_dir}/hpx/hpx_git_log.txt

#    cp ~/src/hpx/build_release_clang_no_hpxmp/hpx_cmake_log.txt ${results_dir}/hpx/hpx_cmake_log.txt
fi

benchmarks_dir=${blazemark_dir}"/benchmarks"

rm -rf ${benchmarks_dir}/build_log.txt
rm -rf ${benchmarks_dir}/compile_log.txt
touch ${benchmarks_dir}/build_log.txt
touch ${benchmarks_dir}/compile_log.txt


cd ${blazemark_dir}
for b in ${benchmarks[@]}
    do
    for r in ${runtimes[@]}
        do
echo $b
echo $r
        rm -rf ${benchmarks_dir}/${b}_${r}
        echo ${b} ${r}>>${benchmarks_dir}/build_log.txt
        echo ${b} ${r}>>${benchmarks_dir}/compile_log.txt
        
        if [ -f ${blazemark_dir}/bin/${b} ]
        then
            make clean
        fi
        
        rm -rf Makefile
        ${blazemark_dir}/configure ${config_dir}/Configfile_${r}>>${benchmarks_dir}/compile_log.txt
        make ${b}>>${benchmarks_dir}/build_log.txt
        mv ${blazemark_dir}/bin/${b} ${benchmarks_dir}/${b}_${r}
        echo "benchmark "${b}" for "${r}" runtime created"
    done
done


