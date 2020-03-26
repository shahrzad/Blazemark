#!/bin/bash
node_name=$(squeue -h -o "%N" -u sshirzad)
node_name=${node_name:0:-2}
echo $node_name
~/scripts/build_scripts/hpx_build_script.sh release gcc no_hpxmp $node_name

