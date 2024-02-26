# Alzheimer

## Dataset:
Please download the dataset from below link:

https://adni.loni.usc.edu/

## Step1: download
source ~/anaconda3/etc/profile.d/conda.sh
##step2: categorize CSV file into subject AD, CN, MCI using below code.


for f in $(find . -type f -name *.nii)
do
	# load IDs from file name
	ptid=$(echo $f | cut -d "/" -f 3)
	name=$(basename $f | cut -d "." -f 1)
	series_id=$(echo $name | rev | cut -d "_" -f 2 | rev | sed 's/S//')
	image_id=$(echo $name | rev | cut -d "_" -f 1 | rev | sed 's/I//')

	# load category from SCV
	screen=$(grep "^.*,${image_id},${ptid},.*,${series_id}$" ADNI_merged.csv | head -1 | cut -d "," -f 4)
	
	# create_directory
	directory_adni=$(echo $f | sed "s/^\.\/ADNI\///")
	directory_base="./categorized_nii/$screen/"
	directory_to="$directory_base$directory_adni"
	mkdir -p $directory_to	
	echo $f $directory_to 
	cp $f $directory_to 
	#med2image -i $f -d $directory_to
done


conda activate med2image
## Step2: Convert nii to jpg    This will produce one
https://www.onlineconverter.com/nifti-to-jpg

# Converted to X axial, Y sagital, Z coronal
## clone the below github project https://github.com/FNNDSC/med2image


## Activate conda environtment the environtment
conda activate med2image
### load IDs or MRI image from file name
make below code a **run_convert.sh**  save it and run it. 

#!/bin/bash
#
source ~/anaconda3/etc/profile.d/conda.sh

conda activate med2image

for f in $(find ./ADNI -type f -name *.nii)
do
	# load IDs from file name
	ptid=$(echo $f | cut -d "/" -f 3)
	name=$(basename $f | cut -d "." -f 1)
	series_id=$(echo $name | rev | cut -d "_" -f 2 | rev | sed 's/S//')
	image_id=$(echo $name | rev | cut -d "_" -f 1 | rev | sed 's/I//')

	# load category from SCV
	screen=$(grep "^.*,${image_id},${ptid},.*,${series_id}$" ADNI_merged.csv | head -1 | cut -d "," -f 4)
	
	# create_directory
	directory_adni=$(echo $f | sed "s/^\.\/ADNI\///")
	directory_base="./tmp/$screen/"
	directory_to="$directory_base$directory_adni"
	mkdir -p $directory_to	

	med2image -i $f -d $directory_to --reslice
done



  

![2](https://github.com/najm-h/Alzheimer/assets/147291760/4085cbf7-14b4-4b97-85e0-3a4295aea7e4)




