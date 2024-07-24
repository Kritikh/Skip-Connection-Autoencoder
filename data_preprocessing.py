import os
import shutil

# Define the path to the folder containing the images
source_folder = 'tid2013/distorted_images'
destination_folder = 'data'

# List all files in the source folder
files = os.listdir(source_folder)

# Loop through all files and process each file
for file_name in files:
    # Check if the file name contains the 'iXX' substring pattern
    if 'i' in file_name and file_name.split('i')[-1][:2].isdigit():
        # Extract the unique substring 'iXX'
        unique_substring = file_name.split('i')[-1][:2]
        folder_name = 'i' + unique_substring
        
        # Create a new folder in the destination path if it doesn't exist
        target_folder = os.path.join(destination_folder, folder_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Copy the file to the new folder
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        shutil.copy(source_path, target_path)

print("Images have been successfully copied into folders.")

