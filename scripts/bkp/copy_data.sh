node="marvin"
benchmark="dmatdmatadd"
cp -r ../results/info-$node-$benchmark ../data/matrix/09-15-2019/
for th in $(seq 1 8)
do 
echo $th
cp  ../results/$node-$benchmark-$th-hpx* ../data/matrix/09-15-2019/
done 


