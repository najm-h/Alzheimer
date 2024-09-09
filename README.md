# Alzheimer

## Dataset:
Please download the dataset from below link:
For ADNI
https://adni.loni.usc.edu/
For MIRIAD 
 https://www.ucl.ac.uk/drc/research-clinical-trials/minimal-interval-resonance-imaging-alzheimers-disease-miriad.

Categorical downloading system: 


For authorized ADNI users only:
Standardized Image Collections:

    Log into the archive: https://ida.loni.usc.edu/login.jsp?project=ADNI
    Under the DOWNLOAD menu, choose “Image Collections”
    In the left navigation section, click “Other Shared Collections”
    Select “ADNI”
    Click on the collection name matching the desired standardized data set
    Download to an appropriately named location on your computer system (i.e. ADNI1_Complete_Year_1)

Standardized Lists:

    Log into the archive: https://ida.loni.usc.edu/login.jsp?project=ADNI
    Under the DOWNLOAD menu, choose “Study Data”
    In the left navigation, click “Study Info” then  “Data & Database”
    Click on the link you wish to download (either ADNI 1.5T MRI Standardized Lists or ADNI 3T MRI Standardized Lists)
    Download to an appropriately named location on your computer system (i.e. ADNI_3T_MRI_Standardized_Lists)

![image](https://github.com/najm-h/Alzheimer/assets/147291760/dbc40566-c4cb-4954-b931-6e8e7863ecc2)

 

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
## Step2:  Converted to X axial, Y sagital, Z coronal
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

########"The full code will be provided soon."


