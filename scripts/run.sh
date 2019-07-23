#!/bin/bash
if [ $# -ne 1 ]
then
echo "node not specified, marvin by default"
node="marvin"
else
node=$1
fi

sbatch -p $node -N 1 --time=72:00:00 ./mat_hpx_chunks.sh $node
