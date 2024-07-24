import os
import shutil
import random

# Define paths
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

# Create train, test, val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List of folders in the data directory
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('i')]

for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    images = os.listdir(folder_path)
    
    # Shuffle the images to ensure random splitting
    random.shuffle(images)
    
    # Calculate the split indices
    total_images = len(images)
    train_split = int(0.8 * total_images)
    test_split = int(0.1 * total_images)
    
    # Split the images
    train_images = images[:train_split]
    test_images = images[train_split:train_split + test_split]
    val_images = images[train_split + test_split:]
    
    # Function to copy images to the respective directories
    def copy_images(image_list, destination_folder):
        dest_folder = os.path.join(destination_folder, folder)
        os.makedirs(dest_folder, exist_ok=True)
        for image in image_list:
            src_image_path = os.path.join(folder_path, image)
            dst_image_path = os.path.join(dest_folder, image)
            shutil.copy(src_image_path, dst_image_path)
    
    # Copy images to train, test, val directories
    copy_images(train_images, train_dir)
    copy_images(test_images, test_dir)
    copy_images(val_images, val_dir)

print("Data has been successfully split and copied to train, test, and val directories.")
