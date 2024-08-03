# Alzheimer

## Dataset:
Please download the dataset from below link:

https://adni.loni.usc.edu/

## Step1: download
## Step2: Convert nii to jpg
https://github.com/FNNDSC/med2image



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
![dowloading_system](https://github.com/musaru/Alzheimer/assets/2803163/2b4a18d6-b162-4d53-8ca2-9f87fefa6217)


%najmul style https://github.com/najm-h/Alzheimer/:
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

#3 Step. Splitting dataset for training and Validation and Testing

X_train, x_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2)

# Number of samples after train test split
print("Number of samples after splitting into Training, validation & test set\n")

print("Train     \t",sorted(Counter(np.argmax(y_train, axis=1)).items()))
print("Validation\t",sorted(Counter(np.argmax(y_val, axis=1)).items()))
print("Test      \t",sorted(Counter(np.argmax(y_test, axis=1)).items()))
# 4 Step. Features Extraction 


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense

class ChannelAttention(Layer):
    def __init__(self, d_model, ratio, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.ratio = ratio
        
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units = d_model//ratio, activation = 'relu')
        self.dense2 = Dense(units = d_model, activation = 'sigmoid')
    
    def build(self, input_shape):
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        dense1 = self.dense1(avg_pool)
        dense2 = self.dense2(dense1)
        dense2 = tf.reshape(dense2, [-1, 1, 1, self.d_model])
        return inputs * dense2
    


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, ReLU, AveragePooling2D, Dropout, Flatten, Dense, Softmax
from tensorflow.keras.initializers import GlorotUniform

#  model
init = GlorotUniform()
model = Sequential()


model.add(Input(shape=(M, N, 3)))

model.add(Conv2D(16, 5, kernel_initializer=init))
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(32, 5, kernel_initializer=init))
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(64, 5, kernel_initializer=init))
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, 5, kernel_initializer=init))
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(256, 5, kernel_initializer=init))
model.add(ReLU())
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(ChannelAttention(256, 5))
model.add(Dropout(0.01))
model.add(Flatten())

model.add(Dense(256, kernel_initializer=init))
model.add(ReLU())
model.add(Dropout(0.03))

model.add(Dense(4, kernel_initializer=init))
model.add(Softmax())


# Print model summary
model.summary()


