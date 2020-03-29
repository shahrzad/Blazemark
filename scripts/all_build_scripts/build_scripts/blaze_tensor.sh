#!/bin/bash
username="sshirzad"
build_dir="/home/${username}/src/blaze_tensor/"
install_dir="/home/${username}/lib/blaze_tensor/"

if [ ${build_dir} != "" ]
then
        rm -rf "${build_dir}/build"
        rm -rf "${build_dir}/build/blazetensor_cmake_log.txt"
        rm -rfv "${install_dir}"

        mkdir "${build_dir}/build"
        mkdir $install_dir
        cd "${build_dir}/build"
        touch blazetensor_cmake_log.txt
        cmake -DCMAKE_INSTALL_PREFIX=${install_dir} -Dblaze_DIR=~/lib/blaze/share/blaze/cmake/ .. | tee blaze_cmake_log.txt
        make install -j 16
else
        echo "Error"
fi

