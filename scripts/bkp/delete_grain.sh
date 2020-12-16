node="marvin"
#results_dir=/home/sshirzad/repos/Blazemark/results_grain_size
result_dir=/home/sshirzad/repos/Blazemark/data/grain_size/${node}

data_dir=/home/sshirzad/repos/Blazemark/data/grain_size/${node}
#cp -r results_dir/info-${node} ${data_dir}
chunk_sizes=(2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)
#iter_lengths=(1 2 3 4 5 6 7 8)
iter_lengths=(50 60 65 70 80 90 100 200 300 400)
num_iterations=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000)


for th in $(seq 1 8)
do 
for c in ${chunk_sizes[@]}
do
for il in ${iter_lengths[@]}	
do
for ni in ${num_iterations[@]}
do
        if [ $c -ne 1 ]
	then
	echo "chunk size ${c}, ${il}, ${ni}"
	rm ${results_dir}/${node}_grain_size_${th}_${c}_${il}_${ni}.dat
	fi
done 
done
done
done

