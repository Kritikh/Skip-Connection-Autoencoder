import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from SSIM_Calculation_Function import ssim
from scipy.io import savemat

def train_model(model, train_loader, val_loader, num_epochs=1, learning_rate=1e-3):
    # Device configuration
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss functions and optimizer
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists('val_results'):
        os.makedirs('val_results')
   
    model.train()
   
    # Training the autoencoder
    for epoch in range(num_epochs):
        for image,label in train_loader:
            img = image.to(device)
            lbl = label.to(device)
            #print(img.shape)
           # print(lbl.shape)
            ssim_map_true = ssim(img, lbl, size_average=False)
           # ssim_map_true = ssim_map_true.cpu().numpy()

            # Forward pass
            output = model(img)
            #output_lbl = model(lbl)
            ssim_map_pred = ssim(output, lbl, size_average=False)
            #print(ssim_map_true.shape)
            #print(ssim_map_pred.shape)
            loss_mse = criterion_mse(ssim_map_pred, ssim_map_true)
            loss_mae = criterion_mae(ssim_map_pred, ssim_map_true)
            loss = loss_mse  # You can choose to optimize for either MSE or MAE
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # Set the model to evaluation mode
        val_loss_mse = 0
        val_loss_mae = 0
        
        with torch.no_grad():
            s=0
            for image,label in val_loader:
                img = image.to(device)
                lbl = label.to(device)
                ssim_map_true = ssim(img, lbl, size_average=False)
                output = model(img)
                ssim_map_pred = ssim(output, lbl, size_average=False)
                #print(ssim_map_true.shape)
                #print(ssim_map_pred.shape)
                loss_mse = criterion_mse(ssim_map_pred, ssim_map_true)
                loss_mae = criterion_mae(ssim_map_pred, ssim_map_true)
                val_loss_mse += loss_mse.item()
                val_loss_mae += loss_mae.item()

                for k in range(len(ssim_map_true)):
                    if not os.path.exists(f'val_results/{epoch+1}'):
                        os.makedirs(f'val_results/{epoch+1}')
                    original_path = os.path.join('val_results', f'{epoch+1}/ssim_map_true_{k+s}.mat')
                    reconstructed_path = os.path.join('val_results', f'{epoch+1}/ssim_map_pred_{k+s}.mat')
                
                    # Save original image
                    # original_img = ssim_map_true[k].cpu().numpy()
                    # original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
                    original_img = np.transpose(ssim_map_true[k].cpu().numpy(), (1, 2, 0))
                    savemat(original_path, {'ssim_map_true': original_img})
                    
                    # Save reconstructed image
                    # reconstructed_img = ssim_map_pred[k].cpu().numpy()
                    # reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
                    reconstructed_img = np.transpose(ssim_map_pred[k].cpu().numpy(), (1, 2, 0))
                    savemat(reconstructed_path, {'ssim_map_pred': reconstructed_img})
                s=s+len(ssim_map_true)
        
        val_loss_mse /= len(val_loader)
        val_loss_mae /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss (MSE): {loss_mse.item():.4f}, Validation Loss (MSE): {val_loss_mse:.4f}, Validation Loss (MAE): {val_loss_mae:.4f}')
        if not os.path.exist(f'weights/{epoch+1}'):
            os.makedirs(f'weights/{epoch+1}')
        torch.save(model.state_dict(), os.path.join('weights',   f'checkpoint_{epoch+1}.pt'))
        
    print('Training finished.')
