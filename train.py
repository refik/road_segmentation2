import os
import pdb
import time
import copy
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


PARAM_ROOT = 'trained_models'


def train_model(model, dataloaders, learning_rate=0.001, num_epochs=50, deeplab=False):
    """Train a given model with given data and save the parameters."""
    
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create a model name, snapshots of the model will be saved under this name
    model_name = time.strftime("%Y%m%d-%H%M%S") + '-net'
    print("Training model with name:", model_name)
    
    # Since we have only one class, using binary cross entropy
    criterion = nn.BCEWithLogitsLoss()
    
    # Initializing the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Time stats
    since = time.time()

    # During the run, best model based on the validation set will be saved
    best_model_wts = None
    best_f1 = 0.0
    
    # Epoch stats so that they can be plotted later
    epoch_stats = {
        'loss': [],
        'f1': [],
        'acc': [],
        'phase': [],
        'epoch': []
    }
    
    for epoch in range(num_epochs): 
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = []
            running_acc = []
            running_pp = 0
            running_cp = 0
            running_tp = 0

            dataloader = dataloaders[phase]

            # Iterate over data.
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)

                    if deeplab:
                        outputs = outputs['out']

                    preds = (torch.sigmoid(outputs) > 0.5) * 1
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    compare = (preds == targets)
                    running_pp += torch.sum(preds == 1).item()
                    running_cp += torch.sum(targets == 1).item()
                    running_tp += torch.sum(compare & (targets == 1)).item()
                    running_acc.append(torch.mean(compare * 1.0).item())
                    running_loss.append(loss.item())

            try:
                # Calculating stats for f1
                prec = running_tp / running_pp
                recall = running_tp / running_cp
                epoch_f1 = 2 * (prec * recall) / (prec + recall)
            except ZeroDivisionError:
                # running_pp or cp might be zero, could improve later
                epoch_f1 = np.nan

            # Recording all stats for the epoch in a dict
            epoch_stats['loss'].append(np.mean(running_loss))
            epoch_stats['f1'].append(epoch_f1)
            epoch_stats['acc'].append(np.mean(running_acc))
            epoch_stats['phase'].append(phase)
            epoch_stats['epoch'].append(epoch)

            print('{} Loss: {:.4f} F1: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_stats['loss'][-1], epoch_f1, epoch_stats['acc'][-1]))

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if epoch % 20 == 0:
            # Take snapshots of the model, could be used for ensambling or debugging
            param_path = os.path.join(PARAM_ROOT, model_name + f'[epoch:{epoch}]' + '.pth')
            stats_path = os.path.join(PARAM_ROOT, model_name + f'[epoch:{epoch}]' + '.sta')
            torch.save(model.state_dict(), param_path)

    # Time and performance information
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the model and epoch statistics
    param_path = os.path.join(PARAM_ROOT, model_name + '.pth')
    stats_path = os.path.join(PARAM_ROOT, model_name + '.sta')
    torch.save(model.state_dict(), param_path)
    pickle.dump(epoch_stats, open(stats_path, 'wb'))

    print(param_path) # Path of the best models params
    print("DONE")
        
    return model, epoch_stats, param_path, best_f1
