#!/bin/bash
username="sshirzad"
build_dir="/home/${username}/src/blaze/"
install_dir="/home/${username}/lib/blaze/"

if [ ${build_dir} != "" ]
then 
	rm -rf "${build_dir}/build"
	rm -rf "${build_dir}/build/blaze_cmake_log.txt"

	mkdir "${build_dir}/build"
	cd "${build_dir}/build"
	touch blaze_cmake_log.txt
	cmake -DBLAZE_SMP_THREADS=C++11 -DCMAKE_INSTALL_PREFIX=${install_dir} .. | tee blaze_cmake_log.txt
	make install -j 16
else
	echo "Error"
fi
