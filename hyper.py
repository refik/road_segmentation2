import os
import time
import pickle
import numpy as np
import pandas as pd

from unet import UNet
from train import train_model
from dataset import get_dataloaders
from visualization import plot_epoch_stats

from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50


TUNE_ROOT = '/tf/projects/road_segmentation/refik/hyper_results'


def tune_hyper(epoch=10, deeplab=False):
    """Tune hyper-parameters for alternative values."""
    
    # All combinations of values here will be tried in a loop
    learning_rate = np.logspace(-3.4, -4.2, 5)
    try_depth = [34]
    random_vflip = [True]
    random_hflip = [True]
    random_rotate = [True]
    random_transform = [True]
    customs = [True, False]
    
    # Model result info
    model_paths = []
    model_best_f1 = []  
    model_vflip = []
    model_hflip = []
    model_transform = []
    model_rotate = []
    model_depth = []
    model_lr = []
    model_custom = []
    
    # Tuning loop
    for md in try_depth:
        for rv in random_vflip:
            for rh in random_hflip:
                for rr in random_rotate:
                    for rt in random_transform:
                        for lr in learning_rate:
                            for custom in customs:
                                # Getting dataset loaders
                                dataloaders = get_dataloaders(
                                    random_transform=rt, 
                                    random_rotate=rr,
                                    random_hflip=rh, 
                                    random_vflip=rv,
                                    all_in=not custom
                                )
                                
                                print(f'Testing for learning rate {lr}')
                                
                                if deeplab:
                                    model = deeplabv3_resnet101(num_classes=1, pretrained=False)
                                else:
                                    model = UNet(n_channels=3, n_classes=1, depth=md)

                                model, epoch_stats, path, best_f1 = train_model(
                                    model, dataloaders, 
                                    num_epochs=epoch, learning_rate=lr, 
                                    deeplab=deeplab
                                )

                                model_paths.append(path)
                                model_best_f1.append(best_f1)
                                model_vflip.append(rv)
                                model_hflip.append(rh)
                                model_transform.append(rt)
                                model_rotate.append(rr)
                                model_depth.append(md)
                                model_lr.append(lr)
                                model_custom.append(custom)

                                print()

    # Collecting in dictionary for a DataFrame
    tune_results = {
        'learning_rate': model_lr,
        'model_depth': model_depth,
        'model_paths': model_paths,
        'random_vflip': model_vflip,
        'random_hflip': model_hflip,
        'random_rotate': model_rotate,
        'random_transform': model_transform,
        'custom': model_custom, 
        'best_f1': model_best_f1
    }
    
    # Saving tuning results for later inspection
    tune_name = time.strftime("%Y%m%d-%H%M%S") + '.tune' 
    tune_path = os.path.join(TUNE_ROOT, tune_name)
    pickle.dump(tune_results, open(tune_path, 'wb'))
    
    print(tune_path)
    print('Tuning complete')
    
    return pd.DataFrame(tune_results), tune_path


def tune_result_epoch_stats(tune_results, idx):
    """Return the epoch statistics of a specific tune result and plot it."""
    model_path = tune_results.iloc[idx].model_paths
    stat_path = model_path.replace('.pth', '.sta')
    epoch_stats = pickle.load(open(stat_path, 'rb'))
    plot_epoch_stats(epoch_stats)
    return epoch_stats

