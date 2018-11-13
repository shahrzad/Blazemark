#!/bin/bash
rm -rf hpx_results/*

thr=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
for th in "${thr[@]}" 
do
	touch hpx_results/hpx_test_${th}.txt
	~/Workspace/hpx_test/build_release/hpx_test --hpx:threads=${th} --hpx:print-counter=/threads/idle-rate --hpx:print-counter=/threads/time/average --hpx:print-counter=/threads/time/average-overhead>>hpx_results/hpx_test_${th}.txt
echo "finished for ${th} threads"
done
