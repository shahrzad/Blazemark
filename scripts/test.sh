#!/bin/bash
#bc -l
#for (( k = 0; k < 50; ++k )); do
#  a=$(( 2*k + 1 ))
#  echo "$a"
#done

#blaze_dir="/home/sshirzad/src/blaze_shahrzad"
#filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h
#
#cache_size=128
#old_line=$(sed -n '118 p' "${filename}")
#old_line=$( echo ${old_line:30:-4} )
#old_part="(${old_line}UL)"
#new_part="(${cache_size}UL)"
#sed -i "118s/$old_part/$new_part/" $filename
#
#echo $(sed -n '118 p' "${filename}")
cd ../data/matrix/dmatdmatmult/reference/12-10-18-0935/
for filename in *
do
echo $filename
#mv "$filename" "${filename%"-10-256"}"
#echo $filename
done
