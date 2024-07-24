import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# Paths to the image folder and the destination folders for training and validation sets
image_folder = 'tid2013/ssim_maps_validate'
train_folder = 'tid2013/ssim_maps_val'
validation_folder = 'tid2013/ssim_maps_test'

# Create directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Get all image paths
image_paths = glob.glob(os.path.join(image_folder, '*.*'))

# Split the dataset into training and validation sets
train_paths, validation_paths = train_test_split(image_paths, test_size=0.5, random_state=42)

# Copy images to the training folder
for img_path in train_paths:
    shutil.copy(img_path, train_folder)

# Copy images to the validation folder
for img_path in validation_paths:
    shutil.copy(img_path, validation_folder)

print(f"Total images: {len(image_paths)}")
print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(validation_paths)}")


