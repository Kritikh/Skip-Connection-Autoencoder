import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from SSIM_Calculation_Function import ssim
from scipy.io import savemat

def visualize_results(model, test_loader, save_dir='results'):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ssim_maps_true = []
    ssim_maps_pred = []
    # Get a batch of data from the test loader
    with torch.no_grad():
        for image, label in test_loader:
            img = image.to(device)
            lbl = label.to(device)
            ssim_map_true = ssim(img, lbl, size_average=False)
            ssim_map_true = ssim_map_true.cpu().numpy()

            
            # Get reconstructed images from the model
            output = model(img)
            ssim_map_pred = ssim(output, lbl, size_average=False)
            ssim_map_pred = ssim_map_pred.cpu().numpy()
            ssim_maps_true.extend(ssim_map_true)
            ssim_maps_pred.extend(ssim_map_pred)


    
    ssim_maps_true = np.asarray(ssim_maps_true)
    ssim_maps_pred = np.asarray(ssim_maps_pred)
    # Save original and reconstructed images
    n = len(ssim_maps_true)
    for i in range(n):
        true_path = os.path.join(save_dir, f'ssim_map_true_{i}.mat')
        pred_path = os.path.join(save_dir, f'ssim_map_pred_{i}.mat')
        
        # Save original image
        # original_img = ssim_map_true[i]
        # original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        original_img = np.transpose(ssim_maps_true[i], (1, 2, 0))

        savemat(true_path, {'ssim_map':original_img})
        
        # Save reconstructed image
        # reconstructed_img = ssim_map_pred[i]
        # reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
        reconstructed_img = np.transpose(ssim_maps_pred[i], (1, 2, 0))
        
        savemat(pred_path, {'ssim_map':reconstructed_img})

    print(f"Images saved to '{save_dir}'")

