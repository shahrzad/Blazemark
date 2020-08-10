#!/bin/bash
#bc -l
#echo $(python -c "from math import ceil;print ($mat_sizemod2)")
#string='(1231,1)'
#if [[ $string != *",1"* ]]; then
#  echo "It's there!"
#fi

qs=""
mode="idle"
ctr=""
adaptive=""

# Argument parsing
while getopts "qm:ac" OPTION; do case $OPTION in
    q)
    	qs="qs"
        ;;
    m)
        mode=$OPTARG
        ;;
    a)
        adaptive="adaptive"
        ;;
    c)
        counter="ctr"
        ;;
esac; done

echo "${qs} ${mode} ${adaptive} ${counter}"

#cd ../data/grain_size/marvin/
#for filename in *
#do
#echo $filename
##mv "$filename" "${filename/marvin-/}"
##mv "$filename" "marvin-${filename}"
#
##echo $filename
#done


#NUMBER=501
#p=$(python -c "from math import ceil; print (ceil($NUMBER/500.0))")
#echo "p: $p"
#NUMBER=1501
#python -c "from math import ceil; print (ceil($NUMBER/500.0))"
#l=$(sed -n '92 p' "/home/sshirzad/src/blaze_shahrzad/blazemark/params/dmatdmatadd.prm")
#l=${l:1:-1}
#E="$(echo -e "${l}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
#echo $E
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
#cd ../data/matrix/dmatdmatmult/reference/12-10-18-0935/
#for filename in *
#do
#echo $filename
##mv "$filename" "${filename%"-10-256"}"
##echo $filename
#done
