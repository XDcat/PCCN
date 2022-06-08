
# init file
result_file="result${date}.txt"
echo > out0.log
date > $result_file

cat aas.txt | while read line
do
	aas=$(echo $line | cut -d " " -f1)  # origin mutation
	script=${line#*:} 
	w=$(echo $line | cut -d " " -f 7)  # wild
	m=$(echo $line | cut -d " " -f 8)  # mutation
	idx=$(echo $line | cut -d " " -f 9)  # idx
	res="7a98_A_${w}_${idx}_${m}.ddg"

	# run
	echo "run script ${aas} -> ${res} : ${script}"
	$script < /dev/null
	echo "------ complete: run script ${aas} -> ${res} : ${script}"

	# save result
	echo "$aas $res" >> $result_file
	cat $res >>  $result_file
done
