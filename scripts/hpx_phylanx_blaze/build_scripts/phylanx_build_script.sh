#!/bin/bash
module purge
module load cmake/3.10.2
module load pybind11/2.2.3
module load gperftools/2.7

gcc_version="8.2.0"
clang_version="6.0.1"
boost_version="1.68.0"
username="sshirzad"
clang_dir="/opt/mn/clang/${clang_version}/bin/clang"
clangpp_dir="/opt/mn/clang/${clang_version}/bin/clang++"
hpx_dir="/home/${username}/lib/hpx"
blaze_dir="/home/${username}/lib/blaze/share/blaze/cmake/"
blaze_tensor_dir="/home/${username}/lib/blaze_tensor/share/blaze/cmake/"
CXX_FLAGS="-Dblaze_DIR=${blaze_dir} -DPHYLANX_WITH_BLAZE_TENSOR=ON -DBlazeTensor_DIR=${blaze_tensor_dir}"

if [ $# -eq 3 ]
then
	build_type=$1
	compiler=$2
	hpxmp_type=$3
	###########build_type##################
	if [ $1 == 'Release' ] || [ $1 == 'release' ]
	then
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_BUILD_TYPE='Release'"
	        build_dir="/home/${username}/src/phylanx/build_release"
                install_dir="/home/${username}/lib/phylanx/phylanx_release"
		hpx_dir="${hpx_dir}/hpx_release"
        elif [ $1 == 'Debug' ] || [ $1 == 'debug' ]
	then
                CXX_FLAGS="${CXX_FLAGS} -DCMAKE_BUILD_TYPE='Debug'"
	        build_dir="/home/${username}/src/phylanx/build_debug"
                install_dir="/home/${username}/lib/phylanx/phylanx_debug"
                hpx_dir="${hpx_dir}/hpx_debug"
	elif [ $1 == 'RelWithDebInfo' ] || [ $1 == 'relwithdebinfo' ]
        then
                CXX_FLAGS="${CXX_FLAGS} -DCMAKE_BUILD_TYPE='RelWithDebInfo'"
                build_dir="/home/${username}/src/phylanx/build_relDeb"
                install_dir="/home/${username}/lib/phylanx/phylanx_relDeb"
                hpx_dir="${hpx_dir}/hpx_relDeb"
	else
		echo "Error: Build type not accepted"
		exit
	fi
	################compiler####################
	if [ ${compiler} == 'clang' ] || [ ${compiler} == 'Clang' ]
	then
		module load clang/${clang_version}
		module load boost/${boost_version}-clang${clang_version}-$1
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_C_COMPILER=${clang_dir} -DCMAKE_CXX_COMPILER=${clangpp_dir} -DCMAKE_CXX_FLAGS=-stdlib=libc++"
		hpx_dir="${hpx_dir}_clang"
		build_dir="${build_dir}_clang"
                install_dir="${install_dir}_clang"

	elif [ ${compiler} == 'gcc' ] || [ ${compiler} == 'GCC' ]
        then
		module load gcc/${gcc_version}
		module load boost/${boost_version}-gcc${gcc_version}-$1
                hpx_dir="${hpx_dir}_gcc"
		build_dir="${build_dir}_gcc"
	        install_dir="${install_dir}_gcc"
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_CXX_FLAGS='-march=native'"

	else
		echo "ERROR: Compiler type not accepted"
                exit
	fi
	###############hpxmp type#############################
	if [ ${hpxmp_type} == 'hpxmp' ] || [ ${hpxmp_type} == 'no_hpxmp' ]
        then
                build_dir="${build_dir}_${hpxmp_type}"
                install_dir="${install_dir}_${hpxmp_type}"
		hpx_dir="${hpx_dir}_${hpxmp_type}/lib64/cmake/HPX/"
        else
                echo "ERROR: hpxmp type not accepted"
                exit
        fi

	if [ ${build_dir} != "" ] && [ ${install_dir} != "" ]
	then 
		rm -rf "${build_dir}"
                rm -rf "${install_dir}"
		rm -rf "${build_dir}/phylanx_cmake_log.txt"
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_INSTALL_PREFIX=${install_dir}"
	else
		echo "ERROR: Build or install directory was not set properly"
		exit
	fi
	
	mkdir ${build_dir}
        mkdir ${install_dir}
	touch "${build_dir}/phylanx_log.txt"
	cd ${build_dir}
	CXX_FLAGS="${CXX_FLAGS} -DHPX_DIR=${hpx_dir}"
	cmake ${CXX_FLAGS} .. | tee phylanx_cmake_log.txt
	make -j 16
	make install -j 16
        make tests -j 16
else
	echo "Error: Three arguments should be provided: build_type compiler hpxmp/no_hpxmp"
fi
