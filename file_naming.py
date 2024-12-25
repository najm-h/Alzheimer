import glob
import re
import os
import shutil

path = '/home/najm/ADNI_musa/nii_image_resized/MCI/////.nii/x/.*'
save_directory = '/home/najm/ADNI_musa/nii_image_resized/MCI_slices'
print(path)

files = glob.glob(path)

for f in sorted(files):
    print()
    # Extract parts from the file path
    disease_type = f.split('/')[5]
    patient_id = f.split('/')[9]
    mri = f.split('/')[10]
    mri_id = ''.join(mri.split('')[:4])
    
    # Extract the file name
    file_name = os.path.basename(f)
    print(file_name)
    
    # Initialize new_file_name
    new_file_name = None
    
    # Determine the new file name based on the content
    if '083' in file_name:
        new_file_name = f"{disease_type}{patient_id}{mri_id}_83.jpg"
    elif '084' in file_name:
        new_file_name = f"{disease_type}{patient_id}{mri_id}_84.jpg"
    
    # Save the file if a new name was determined
    if new_file_name:
        new_file_path = os.path.join(save_directory, new_file_name)
        shutil.copy(f, new_file_path)
        print(f"Saved {file_name} as {new_file_name} in {save_directory}")



%%%
path='D:/JOB/Eyes Japan/Pakrinson Project/Company_Data//_1*.*'

files = glob.glob(path)
print(files)
for f in sorted(files):
    #(f.replace('\\', '/')).split('/')[-5].split('_')[1]
    #print(f)
    subject=(f.replace('\\', '/')).split('/')[-2]
    #print(subject)
    task_string=(f.replace('\\', '/')).split('/')[-1].split('_')[1]
    #print(task_string)
    tm=task_string[:2]


%%%
