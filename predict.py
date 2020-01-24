import os
import pdb
import torch
import numpy as np

from PIL import Image
from dataset import get_testloader
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF


IMAGE_PATH = 'predictions/'


def write_predictions(model, model_path, image_path=IMAGE_PATH, patched=False, deeplab=False, 
                      loader_fn=get_testloader, train_img=False, pad=False):
    """Write the predictions from the test data given an empty model and parameter set."""
    
    # Load the model from the given path
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Get the test images
    testloader = loader_fn(pad=pad)
    
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for idx, image in enumerate(testloader):
            if train_img:
                image = image[0].to(device)
            else:
                image = image.to(device)
                
            # Get the prediction of the model
            pred = model(image)
            
            if deeplab:
                pred = pred['out']
            
            # Convert to probability and then binary
            pred = pred.cpu()
            pred = torch.sigmoid(pred).numpy()
            pred = (pred > 0.35) * 1
            pred = pred.astype(np.uint8) * 255
            
            if patched:
                pred = pred.repeat(16, axis=2).repeat(16, axis=3)
            
            # Convert to PIL Image
            pred = Image.fromarray(pred[0][0])
            
            if pad:
                pred = TF.center_crop(pred, (608, 608))
            
            # Write to folder
            path = os.path.join(image_path, f'prediction_{(idx+1):03}.png')
            pred.save(path)
            
    print("OK")
