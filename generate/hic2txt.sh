#!/bin/bash

if [ $# -ne 3 ];
then
    echo "The parameter should be 3(juicer_tools_path, hic_path, output_folder)"
    exit
fi
juicer_tools_path=$1
hic_path=$2
output_folder=$3

for ((i=1; i<=22; i++))
do
	mkdir -p $output_folder
	java -jar $juicer_tools_path dump observed VC $hic_path $i $i BP 1000 $output_folder/chr${i}_1000b.txt
done
