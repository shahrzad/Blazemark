#!/bin/bash
build_dir="/home/sshirzad/src/hpx/build_release_clang_no_hpxmp_marvin_old"
hpx_source_dir="/home/sshirzad/src/hpx"
cd $hpx_source_dir
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "$BRANCH branch">>${build_dir}/info/hpx_git_logi_1.txt
git --git-dir $hpx_source_dir/.git log>>${build_dir}/info/hpx_git_log_1.txt

