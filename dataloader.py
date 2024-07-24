import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# class TID2013Dataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_dir = image_dir
#         self.image_paths = glob.glob(os.path.join(image_dir, '*.bmp'))  # Change the extension if necessary
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path)
#         if self.transform:
#             image = self.transform(image)
#             # plt.imsave('img.jpg', image.cpu().numpy())
#         return image

class TID2013Dataset(Dataset):
    def __init__(self, label_dir, image_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_pathss = glob.glob(os.path.join(label_dir, '*.bmp'))
        self.image_paths=[]
        self.label_paths=[]

        for label in self.label_pathss :
            image_paths = glob.glob(os.path.join(image_dir, os.path.splitext(os.path.basename(label))[0], '*.bmp'))
           # print(label)
           # print(np.asarray(image_paths).shape[0])
            label_paths = [label]*np.asarray(image_paths).shape[0]
            self.label_paths.append(np.array(label_paths))
            self.image_paths.append(np.array(image_paths))
        print(np.asarray(self.label_paths).shape)
        print(np.asarray(self.image_paths).shape)
        self.label_paths = np.array(self.label_paths).flatten()
        self.image_paths = np.array(self.image_paths).flatten() 

        print(np.array(self.label_paths).shape)
        print(np.array(self.image_paths).shape)
        
        self.transform = transform

    def __len__(self):
        print(len(self.image_paths))
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        print(image_path, ' ', label_path)
        image = Image.open(image_path)
        label = Image.open(label_path)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)# plt.imsave('img.jpg', image.cpu().numpy())
        return image,label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((192, 256)),
    transforms.ToTensor(),
])

# Load the TID2013 dataset
def get_dataloaders(train_image_dir = 'data/train' ,test_image_dir = 'data/test',val_image_dir='data/val',org_image_dir='data/reference_images',batch_size=10):
    train_dataset = TID2013Dataset(image_dir=train_image_dir,label_dir=org_image_dir, transform=transform)
    test_dataset = TID2013Dataset(image_dir=test_image_dir, label_dir=org_image_dir, transform=transform)
    validate_dataset = TID2013Dataset(image_dir=val_image_dir,label_dir=org_image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(validate_dataset, batch_size = 128,shuffle=False)
    return train_loader, test_loader, val_loader