nodes=("marvin")
benchmarks=("dmatdmatadd" "dmatdmatdmatadd")
rows=(4)

for node in ${nodes[@]}
do
for benchmark in ${benchmarks[@]}
do
for th in $(seq 1 20)
do
#for r in ${rows[@]}
#do
#rm ../results/$node-$benchmark-$th-hpx-*-$r-*
rm  ../results/$node-$benchmark-$th-hpx* 
#done 
done
done
done
