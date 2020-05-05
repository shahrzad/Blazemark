#!/bin/bash
module purge
module load cmake/3.16.4

#module load cmake/3.10.2
#module load gperftools/2.7
#module load papi/5.7.0

gcc_version="9.1.0"
clang_version="6.0.1"
boost_version="1.68.0"
#clang_version="8.0.0"
#boost_version="1.70.0"
username="sshirzad"
clang_dir="/opt/apps/clang/${clang_version}/bin/clang"
clangpp_dir="/opt/apps/clang/${clang_version}/bin/clang++"
#clang_dir="/opt/mn/clang/${clang_version}/bin/clang"
#clangpp_dir="/opt/mn/clang/${clang_version}/bin/clang++"
CXX_FLAGS="-DHPX_WITH_PAPI=ON -DHPX_WITH_THREAD_IDLE_RATES=ON -DHPX_WITH_THREAD_LOCAL_STORAGE=ON -DHPX_WITH_DYNAMIC_HPX_MAIN=OFF"
if [ $# -eq 4 ]
then
	build_type=$1
	compiler=$2
	hpxmp_type=$3
	node_name=$4
	####################build type########################
	if [ $1 == 'Release' ] || [ $1 == 'release' ]
	then
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_BUILD_TYPE='Release'"
	        build_dir="/home/${username}/src/hpx/build_release"
                install_dir="/home/${username}/lib/hpx/hpx_release"
		build_type="release"
        elif [ $1 == 'Debug' ] || [ $1 == 'debug' ]
	then
                CXX_FLAGS="${CXX_FLAGS} -DCMAKE_BUILD_TYPE='Debug' -DHPX_WITH_THREAD_BACKTRACE_DEPTH=50"
	        build_dir="/home/${username}/src/hpx/build_debug"
                install_dir="/home/${username}/lib/hpx/hpx_debug"
		build_type="debug"
	elif [ $1 == 'RelWithDebInfo' ] || [ $1 == 'relwithdebinfo' ] || [ $1 == 'relDeb' ] || [ $1 == 'reldeb' ]
        then
                CXX_FLAGS="${CXX_FLAGS} -DCMAKE_BUILD_TYPE='RelWithDebInfo'"
		build_dir="/home/${username}/src/hpx/build_relDeb"
                install_dir="/home/${username}/lib/hpx/hpx_relDeb"
		build_type="release"
	else
		echo "Error: Build type not accepted"
		exit
	fi
        ####################compiler type########################
	if [ ${compiler} == 'clang' ] || [ ${compiler} == 'Clang' ]
	then
                module load clang/${clang_version}
#                module load boost/${boost_version}-clang${clang_version}-$build_type
                module load boost/${boost_version}-clang6-$build_type

		build_dir="${build_dir}_clang"
                install_dir="${install_dir}_clang"
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_C_COMPILER=${clang_dir} -DCMAKE_CXX_COMPILER=${clangpp_dir} -DCMAKE_CXX_FLAGS=-stdlib=libc++"
	elif [ ${compiler} == 'gcc' ] || [ ${compiler} == 'GCC' ]
        then
		module load gcc/${gcc_version}
                module load boost/${boost_version}-gcc${gcc_version}-$1

                build_dir="${build_dir}_gcc"
                install_dir="${install_dir}_gcc"
	else
		echo "ERROR: Compiler type not accepted"
                exit
	fi
        ####################hpxmp type########################

	if [ ${hpxmp_type} == 'hpxmp' ] || [ ${hpxmp_type} == 'no_hpxmp' ]
        then
                build_dir="${build_dir}_${hpxmp_type}"
                install_dir="${install_dir}_${hpxmp_type}"
		if [ ${hpxmp_type} == 'hpxmp' ]
		then
	                CXX_FLAGS="${CXX_FLAGS} -DHPX_WITH_HPXMP=ON -DHPX_WITH_HPXMP_NO_UPDATE=ON"
		fi
        else
                echo "ERROR: hpxmp type not accepted"
                exit
        fi

        build_dir="${build_dir}_${node_name}_old"
        install_dir="${install_dir}_${node_name}_old"

	if [ ${build_dir} != "" ] && [ ${install_dir} != "" ]
	then 
		rm -rf "${build_dir}"
                rm -rf "${install_dir}"
		rm -rf "${build_dir}/hpx_cmake_log.txt"
		CXX_FLAGS="${CXX_FLAGS} -DCMAKE_INSTALL_PREFIX=${install_dir}"
	else
		echo "ERROR: Build or install directory was not set properly"
		exit
	fi
 
	mkdir ${build_dir}
        mkdir ${build_dir}/info
	mkdir ${install_dir}
	touch "${build_dir}/info/hpx_cmake_log.txt"
        touch "${build_dir}/info/hpx_git_log.txt"

	cd ${build_dir}

	cmake ${CXX_FLAGS} .. | tee ${build_dir}/info/hpx_cmake_log.txt
	echo "Running on $node">>${build_dir}/info/hpx_cmake_log.txt
	make -j 16
	make install -j 16

	hpx_source_dir="/home/sshirzad/src/hpx"
	cd $hpx_source_dir
	BRANCH=$(git rev-parse --abbrev-ref HEAD)
	echo "$BRANCH branch">>${build_dir}/info/hpx_git_log.txt
	git --git-dir $hpx_source_dir/.git log>>${build_dir}/info/hpx_git_log.txt
else
	echo "Error: Three arguments should be provided: build_type compiler hpxmp/no_hpxmp"
fi
