#!/bin/bash
blaze_dir=/home/sshirzad/src/blaze_shahrzad/
if [ $1 == 'reset' ]
then
    if [ $2 == 'HPX.h' ]
    then
        filename=${blaze_dir}/blaze/config/HPX.h
    elif [ $2 == 'DendeVector.h' ]
    then 
        filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h
    fi
    git checkout ${filename}
    echo "$2 file was reset to original"
elif [ $# -eq 2 ]
then 
    if [ $1 == 'CACHE_SIZE' ]
        then
           filename=${blaze_dir}/blaze/math/smp/hpx/DenseVector.h

           cache_size=$2
           old_line=$(sed -n '118 p' "${filename}")
           old_line=$( echo ${old_line:30:-4} )
           old_part="(${old_line}UL)"
           new_part="(${cache_size}UL)"
           sed -i "118s/$old_part/$new_part/" $filename
           new_line=$(sed -n '118 p' "${filename}")
           new_line=$( echo ${new_line:30:-4} )

           echo $1 "changed from" ${old_line} "to" ${new_line}
    else
	if [ $1 == 'BLAZE_HPX_VECTOR_THRESHOLD' ]
	    then
	        line_number=37
	
	elif [ $1 == 'BLAZE_HPX_VECTOR_CHUNK_SIZE' ]
	    then 
	        line_number=41
	
	elif [ $1 == 'BLAZE_HPX_VECTOR_THREADS_MULTIPLYER' ]
	    then
	        line_number=45
        fi
        filename=${blaze_dir}/blaze/config/HPX.h
	param=$(sed -n ${line_number}' p' ${filename} |cut -d' ' -f3)
	sed -i ${line_number}'s/.*/#define '$1' '$2'/' ${filename}
	echo $1 "changed from" ${param} "to" $(sed -n ${line_number}' p' ${filename}|cut -d' ' -f3)
    fi
fi

