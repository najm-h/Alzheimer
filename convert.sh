#!/bin/bash

# Split ADNI dataset by diagnosis group and convert into jpg images

CSV_FILE="./ADNI1_Annual_2_Yr_3T_1_26_2022.csv"
OUTPUT_DIR="./ADNI_CONVERTED"

source ~/anaconda3/etc/profile.d/conda.sh

conda activate med2image

for f in $(find ./ADNI -name "*.nii" -type f); do
    base=$(basename ${f%.nii})
    IFS='_' read -r -a array <<< "$base"
    
    subject_id="${array[1]}_${array[2]}_${array[3]}"
    modality="${array[4]}"
    description=$(echo $base | sed -e "s/^.*${modality}_//g" -e "s/_Br_.*\$//g")
    imageid="${array[-1]}"

    col=$(grep -e "$subject_id" $CSV_FILE | grep -e "$imageid")
    group=$(echo $col | cut -d, -f 3 | sed -e "s/\"//g")

    prefix=$OUTPUT_DIR/$group/$subject_id/$description_$imageid

    mkdir -p $prefix
    med2image -i $f -d $prefix --reslice

    for i in $(find $prefix -type f -name *.jpg); do
        convert $i -resize 256x256\! $i
    done

done