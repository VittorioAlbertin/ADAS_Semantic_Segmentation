import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

from src.config import DATASET_ROOT, CROP_SIZE, MEAN, STD, IGNORE_INDEX

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = root
        self.split = split
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.target_type = target_type
        self.transform = transform
        
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        self.targets_dir = os.path.join(self.root, self.mode, self.split)
        
        self.images = []
        self.targets = []
        
        # Walk through directory
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            
            for file_name in os.listdir(img_dir):
                if file_name.endswith('.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = file_name.replace('leftImg8bit.png', f'{self.mode}_labelIds.png')
                    self.targets.append(os.path.join(target_dir, target_name))

        # Official mapping from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        # We map id to trainId
        self.id_to_trainId = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]) # Load as PIL (indices)

        # Joint Transform (Random Crop) if training
        if self.split == 'train':
            # Random Crop logic manually to ensure same crop for img and target
            # Note: T.RandomCrop handles PIL images
            i, j, h, w = T.RandomCrop.get_params(image, output_size=CROP_SIZE)
            image = T.functional.crop(image, i, j, h, w)
            target = T.functional.crop(target, i, j, h, w)

        # Mapping: Apply id -> trainId strictly
        target_np = np.array(target, dtype=np.uint8)
        mask = np.full_like(target_np, IGNORE_INDEX)
        for k, v in self.id_to_trainId.items():
            mask[target_np == k] = v
        
        target = torch.from_numpy(mask).long()

        # Image Transform (Normalization)
        transform_img = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        image = transform_img(image)

        return image, target
