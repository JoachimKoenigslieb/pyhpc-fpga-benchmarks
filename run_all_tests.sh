folders="baseline fixedpoint packed pipeline"
target="sw_emu"
echo "running tests... $folders"
while IFS="" read -r line || [ -n "$p" ]
do
	inputs=(${line})	
	for i in $folders
	do
		cd "$i"/add
		echo "in dir $(pwd)"
		echo "running command: ./host -xclbin ${target}_kernels.xclbin -size ${inputs[3]} -X ${inputs[0]} -Y ${inputs[1]} -Z ${inputs[2]} 5 times"
		(/usr/bin/time -p ./host -xclbin ${target}_kernels.xclbin -size ${inputs[3]} -X ${inputs[0]} -Y ${inputs[1]} -Z ${inputs[2]}) &> output.txt
		for j in {1..4}
		do
			echo "running ${j}th iteration..."
			echo "" >> output.txt
			(/usr/bin/time -p ./host -xclbin ${target}_kernels.xclbin -size ${inputs[3]} -X ${inputs[0]} -Y ${inputs[1]} -Z ${inputs[2]}) &>> output.txt
		done
		cp output.txt ../../runs/${i}_${inputs[3]}.txt
		cd ../..
	done
done < available_sizes
