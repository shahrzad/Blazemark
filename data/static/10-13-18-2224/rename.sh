#!/bin/bash
for FILENAME in *; do
       if [[ $FILENAME == daxpy* ]] && [[ $FILENAME != *openmp* ]]
       then
	a=$(echo $FILENAME | cut -d'-' -f1)'-'$(echo $FILENAME | cut -d'-' -f2)'-'$(echo $FILENAME | cut -d'-' -f3)'-8-'$(echo $FILENAME | cut -d'-' -f4)
       echo $a	;
       mv $FILENAME $a;
fi
done

#$ for FILENAME in *; do mv $FILENAME $FILENAME; done
