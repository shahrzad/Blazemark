#!/bin/bash
module purge
module load cmake/3.9.0
module load clang/6.0.1
module load boost/1.68.0-clang6.0.1-release
module load gperftools/2.7

hpxmp_dir="/home/sshirzad/src/hpxMP"
hpx_dir="/home/sshirzad/lib/hpx/hpx_release_clang_idle_no_hpxmp/lib64/cmake/HPX"
rm -rf "$hpxmp_dir/build_release_idle"
mkdir "$hpxmp_dir/build_release_idle"
cd "$hpxmp_dir/build_release_idle"
touch "$hpxmp_dir/build_release_idle/hpxmp_log.txt"

cmake -DCMAKE_BUILD_TYPE='Release' -DCMAKE_C_COMPILER=/opt/mn/clang/6.0.1/bin/clang -DCMAKE_CXX_COMPILER=/opt/mn/clang/6.0.1/bin/clang++ -DCMAKE_CXX_FLAGS=-stdlib=libc++ -DHPX_DIR=$hpx_dir .. | tee hpxmp_cmake_log.txt
make -j 16
