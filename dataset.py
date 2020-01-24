import os
import numpy as np
import pdb

import torch
from PIL import Image
from skimage import io
import torchvision.transforms.functional as TF
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torch.utils.data import DataLoader


TRAIN_PATH = 'data/training'
TEST_PATH = 'data/test_set_images'
TRAIN_SIZE = 100


class SatelliteDataset(VisionDataset):
    """Satellite images for road classification."""
        
    def __init__(self, root, builtin_transform=True, random_transform=True,
                 random_hflip=False, random_vflip=False, random_rotate=False,
                 pixel_threshold=0.5, indices=np.arange(100), patched_target=False):
        
        super(SatelliteDataset, self).__init__(root)
        
        self.pixel_threshold = pixel_threshold
        self.builtin_transform = builtin_transform
        self.indices = indices
        self.random_transform = random_transform
        self.random_rotate = random_rotate
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.patched_target = patched_target
        self.data = []
        self.targets = []

        # Loading images
        img_dir = os.path.join(self.root, 'images')
        target_dir = os.path.join(self.root, 'groundtruth')
        self.img_count = len(self.indices)
        
        for idx in self.indices:
            img = io.imread(os.path.join(img_dir, f'satImage_{(idx+1):03}.png'))
            target = io.imread(os.path.join(target_dir, f'satImage_{(idx+1):03}.png'))
            
            self.data.append(img)
            
            if self.patched_target == False:
                # Transforming target pixels to 0 and 1 based on a ratio.
                # Multiplying by 255 is because the target image is uint8
                # This is not necessary if the target will be patched
                target = target > (self.pixel_threshold * 255)

            self.targets.append(target)
        
        print(f'Loaded {self.img_count} images.')
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        

    def transformed(self, img, target):
        """Return transformed images for data augmentation."""
        if self.random_transform:
            if self.random_rotate:
                # Random rotate
                degree = np.random.randint(0, 360)
                img = TF.pad(img, 120, padding_mode='reflect')
                img = TF.rotate(img, degree)
                
                # Random crop
                img = TF.center_crop(img, 450)
                i, j, h, w = transforms.RandomCrop.get_params(
                    img, output_size=(312, 312))
                img = TF.crop(img, i, j, h, w)
                
                # Do the same for the target
                target = TF.pad(target, 120, padding_mode='reflect')
                target = TF.rotate(target, degree)
                target = TF.center_crop(target, 450)
                target = TF.crop(target, i, j, h, w)

            if self.random_hflip:
                # Random horizontal flipping
                if np.random.random() > 0.5:
                    img = TF.hflip(img)
                    target = TF.hflip(target)

            if self.random_vflip:
                # Random vertical flipping
                if np.random.random() > 0.5:
                    img = TF.vflip(img)
                    target = TF.vflip(target)      
            
        # Transform to tensor
        img = TF.to_tensor(img)
        target = TF.to_tensor(target)
        
        # Normalize image
        img = TF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        if self.patched_target:
            # Decreasing resolution of target
            target = patch_target(target)
        
        return img, target
        
        
    def __len__(self):
        return self.img_count
        
        
    def __getitem__(self, idx):
        """Return a specific image."""
        img = Image.fromarray(self.data[idx])
        target = Image.fromarray(self.targets[idx])
        
        if self.builtin_transform:
            img, target = self.transformed(img, target)
        
        return img, target

    
class SatelliteTestset(VisionDataset):
    """Satellite test set images for road classification submission."""
        
    def __init__(self, root, transform, pad=False):
        
        super(SatelliteTestset, self).__init__(root, transform=transform)
        
        self.data = []
        self.pad = pad
        self.img_count = len(os.listdir(self.root))
        
        for idx in range(self.img_count):
            img = io.imread(os.path.join(root, f'test_{idx+1}', f'test_{idx+1}.png'))
            self.data.append(img)
        
        print(f'Loaded {self.img_count} images.')
        
        self.data = np.array(self.data)
        
        
    def __len__(self):
        return self.img_count
        
        
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        
        if self.pad:
            img = TF.pad(img, 120, padding_mode='reflect')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    
def get_dataloaders(train_ratio=0.8, random_transform=False, seed=1337,
                    random_hflip=False, random_vflip=False, random_rotate=False, 
                    patched_target=False, all_in=False):
    """Return train and validation data loaders (batched, shuffled) for training."""
    dataset_size = TRAIN_SIZE
    np.random.seed(seed)
    
    # Select a random portion of training images for validation
    train_ind = np.random.choice(np.arange(dataset_size), replace=False,
                                 size=int(dataset_size * train_ratio))
    val_ind = np.setdiff1d(np.arange(dataset_size), train_ind) 

    if all_in:
        train_ind = np.arange(100)
    
    # Training set
    trainset = SatelliteDataset(
        root=TRAIN_PATH, 
        indices=train_ind, 
        random_transform=random_transform, 
        random_hflip=random_hflip, 
        random_vflip=random_vflip, 
        random_rotate=random_rotate, 
        patched_target=patched_target
    )

    # Validation set
    valset = SatelliteDataset(
        root=TRAIN_PATH, 
        indices=val_ind, 
        random_transform=False 
    )

    dataloaders = {
        'train': DataLoader(trainset, batch_size=6, shuffle=True, num_workers=2),
        'val': DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)
    }
    
    return dataloaders


def get_testloader(path=TEST_PATH, pad=False):
    """Return loaders for submission test set."""
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = SatelliteTestset(
        root=path, 
        transform=transform,
        pad=pad
    )
    
    return DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

