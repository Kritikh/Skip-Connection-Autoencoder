import os
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from SSIM_Calculation_Function import ssim
import torch
from torchvision import transforms


# Directories for the original and distorted images
original_images_dir = 'tid2013/reference_images'
distorted_images_dir = 'tid2013/distorted_images'
output_ssim_maps_dir = 'tid2013/ssim_maps'


# Ensure the output directory exists
os.makedirs(output_ssim_maps_dir, exist_ok=True)

# Function to load images from a directory
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith('.bmp'):
            img = io.imread(os.path.join(folder, filename))
            if img is not None:
                images[filename] = img
    return images

# Load original and distorted images
original_images = load_images_from_folder(original_images_dir)
distorted_images = load_images_from_folder(distorted_images_dir)

# Function to find distorted images corresponding to an original image
def find_distorted_images(original_filename, distorted_images):
    base_name = os.path.splitext(original_filename)[0].lower()
    print(base_name)
    related_images = {name: img for name, img in distorted_images.items() if base_name in name}
    return related_images

# Function to compute and save SSIM map
def compute_and_save_ssim(original_img, distorted_img, distorted_filename):
    # # Determine the smallest dimension of the images and set the window size accordingly
    # min_dim = min(original_img.shape[:2])
    # win_size = min(7, min_dim)  # Use a window size of at most 7x7, or smaller if the image is smaller
    
    # Compute SSIM map
    ssim_map = ssim(original_img, distorted_img, size_average=False)

    
    ssim_map_image = transforms.ToPILImage()(ssim_map.squeeze(0).cpu())
    #print(ssim_map.squeeze(0).shape)

    # Save the SSIM map
    output_path = os.path.join(output_ssim_maps_dir, f'ssim_map_{distorted_filename}')
    ssim_map_image.save(output_path)
    
    
    # # Save the SSIM map
    
    # io.imsave(output_path, ssim_map_uint8)

# Process each original image and its corresponding distorted images
for original_filename, original_img in original_images.items():
    related_distorted_images = find_distorted_images(original_filename, distorted_images)
    for distorted_filename, distorted_img in related_distorted_images.items():
        compute_and_save_ssim(torch.from_numpy(original_img).unsqueeze(0), torch.from_numpy(distorted_img).unsqueeze(0), distorted_filename)