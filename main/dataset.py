import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class SaliencyDataset(Dataset):
    """
    Standard custom PyTorch Dataset class that handles loading and preprocessing 
    of pristine image and fixation map data for baseline training and inference.
    """
    def __init__(self, data_dir, images_txt, fixations_txt=None):
        self.data_dir = data_dir
        
        with open(os.path.join(data_dir, images_txt), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
            
        if fixations_txt:
            with open(os.path.join(data_dir, fixations_txt), 'r') as f:
                self.fixation_files = [line.strip() for line in f.readlines()]
        else:
            self.fixation_files = None

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.fix_transform = transforms.ToTensor() 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.fixation_files:
            fix_path = os.path.join(self.data_dir, self.fixation_files[idx])
            fixation = Image.open(fix_path).convert('L') 
            
            image = self.img_transform(image)
            fixation = self.fix_transform(fixation)
            return image, fixation
        
        image = self.img_transform(image)
        return image, self.image_files[idx]


class AugmentedSaliencyDataset(SaliencyDataset):
    """
    An extension of SaliencyDataset that applies perfectly synchronized data augmentations
    (horizontal flips and minor rotations) across both inputs and targets during training.
    """
    def __init__(self, data_dir, images_txt, fixations_txt=None, is_training=False):
        super().__init__(data_dir, images_txt, fixations_txt)
        self.is_training = is_training
        # Isolate normalization layer for late processing post-augmentation
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.fixation_files:
            fix_path = os.path.join(self.data_dir, self.fixation_files[idx])
            fixation = Image.open(fix_path).convert('L')
            
            # --- SYNCHRONIZED SPATIAL TRANSFORMATIONS ---
            if self.is_training:
                # 1. Random Horizontal Flip (50% chance)
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    fixation = TF.hflip(fixation)
                
                # 2. Random Subtle Rotation (50% chance, clamped between -10 and +10 degrees)
                if random.random() > 0.5:
                    angle = random.uniform(-10.0, 10.0)
                    image = TF.rotate(image, angle)
                    fixation = TF.rotate(fixation, angle)
            
            # Convert both elements to tensors after augmentations are complete
            image_tensor = transforms.ToTensor()(image)
            image_tensor = self.normalize(image_tensor)
            
            fixation_tensor = transforms.ToTensor()(fixation)
            
            return image_tensor, fixation_tensor
            
        # Test set mode fallback
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)
        return image_tensor, self.image_files[idx]