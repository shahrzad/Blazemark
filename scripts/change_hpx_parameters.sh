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
    if [ $1 == 'BLAZE_HPX_VECTOR_BLOCK_SIZE' ]
            then
            line_number=45
    elif [ $1 == 'BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW' ]
            then
            line_number=53
    elif [ $1 == 'BLAZE_HPX_VECTOR_THRESHOLD' ]
	    then
	        line_number=37
    elif [ $1 == 'BLAZE_HPX_VECTOR_CHUNK_SIZE' ]
	    then 
	        line_number=41
	
    elif [ $1 == 'BLAZE_HPX_MATRIX_CHUNK_SIZE' ]
	    then
	        line_number=49

    elif [ $1 == 'BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN' ]
            then
            line_number=57
    elif [ $1 == 'BLAZE_HPX_SPLIT_TYPE_IDLE' ]
	    then 
	    line_number=61
    elif [ $1 == 'BLAZE_SPLIT_ADAPTIVE' ]
            then
            line_number=69
    fi
        filename=${blaze_dir}/blaze/config/HPX.h
	param=$(sed -n ${line_number}' p' ${filename} |cut -d' ' -f3)
	sed -i ${line_number}'s/.*/#define '$1' '$2'/' ${filename}
	echo $1 "changed from" ${param} "to" $(sed -n ${line_number}' p' ${filename}|cut -d' ' -f3)
fi

