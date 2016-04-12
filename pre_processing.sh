#!/bin/bash

current_path="$(pwd)"
dataset_path=$1
path_to_output=$2
path_to_frame_lists=$3

echo "Current Path: " $current_path
echo "Path to dataset: " $dataset_path
echo "Path to output: " $path_to_output
echo "Path to frame lists" $path_to_frame_lists

mkdir -p $path_to_frame_lists

cd $dataset_path;

echo
echo "List of directories"
for f in *;
	do echo $f;
	mkdir -p $path_to_output/$f;
	for g in $f/*;
  		do echo $dataset_path/$g;
  	done  > $path_to_frame_lists/$f.txt ; 
  	echo "Stabilizing video: " $f
  	echo
  	$current_path/preprocessing/videostab/videostab $path_to_frame_lists/$f.txt $path_to_output $dataset_path 1>>$current_path/stabilization_log.txt
  	echo
done

rm -r $path_to_frame_lists
echo
echo "Creating new Frame lists"
echo
mkdir -p $path_to_frame_lists

for f in *;
	do for g in $f/*;
  		do echo $path_to_output/$g;
  	done  > $path_to_frame_lists/$f.txt ; 
done

cd $current_path



