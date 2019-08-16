#!/bin/bash
module purge
module load cmake/3.10.2
module load clang/6.0.1
module load boost/1.68.0-clang6.0.1-debug
module load gperftools/2.7

hpxmp_dir="/home/sshirzad/src/hpxMP"
hpx_dir="/home/sshirzad/lib/hpx/hpx_debug_clang_no_hpxmp/lib64/cmake/HPX"
rm -rf "$hpxmp_dir/build_debug"
mkdir "$hpxmp_dir/build_debug"
cd "$hpxmp_dir/build_debug"
touch "$hpxmp_dir/build_debug/hpxmp_log.txt"

cmake -DCMAKE_BUILD_TYPE='Debug' -DCMAKE_C_COMPILER=/opt/mn/clang/6.0.1/bin/clang -DCMAKE_CXX_COMPILER=/opt/mn/clang/6.0.1/bin/clang++ -DCMAKE_CXX_FLAGS=-stdlib=libc++ -DHPX_DIR=$hpx_dir -DHPX_WITH_TRACE=ON ..| tee hpxmp_cmake_log.txt
make -j 16
