"""
LFW dataloading
"""
import argparse
import time

import numpy as np
from PIL import Image

import glob
import os

# MATPLOTLIB
import matplotlib.pyplot as plt

# TORCH
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:

        self.transform = transform
        # gather all the images
        folder_path = os.path.join(path_to_folder, "**/*.jpg")
        self.paths = glob.glob(folder_path)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # get paths by index
        path = self.paths[index]
        image = Image.open(path)
        return self.transform(image)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='./lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', default=False, action='store_true')
    parser.add_argument('-get_timing', default=True, action='store_true')
    parser.add_argument('-batches_to_check', default=2, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    if args.visualize_batch:
        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers
        )

        for image in data_loader:

            grid = make_grid(image)
            if not isinstance(grid, list):
                grid = [grid]
            fig, axes = plt.subplots(ncols=len(grid), squeeze=False)
            for (i, image) in enumerate(grid):
                image = image.detach()
                image = F.to_pil_image(image)
                axes[0, i].imshow(np.asarray(image))
            plt.show()

            break

        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing: {np.mean(res)}+-{np.std(res)}')