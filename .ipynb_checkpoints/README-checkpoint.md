# Machine Learning Project 2, Road Segmentation (AICrowd)

**Group Name: evoline2.0**
1. Brighton Muffat
2. Refik Turkeli
3. Samson Zhang

Submission ID: `31396`

## Code Information

Running `run.py` creates the submission on `data/submission.csv` by default. Before running it, the model parameters from the link below has to be downloaded to the `trained_models/` folder. `run.py` has the training code commented out since it is taking around 3 hours with gpu. However, it can be run with a lower epoch value.

Download link: [20191209-020809-net.pth](https://drive.google.com/uc?id=1yIk0AqqCVfneRFNrVeJtFu-LKYCuuFvu&export=download)

### Important Files

- `dataset.py` This file prepares the images for training, validation and ultimately prediction. It initializes the loaders that feed the data to the GPU with threads. It also includes the data augmentation logic. 
- `hyper.py` This file contains the hyper-parameter optimization function. It is basically a loop that tries the combination of arguments given to it. The results are returned as a DataFrame and the best parameters are selected for a more extensive training. Results are also saved in the `hyper_results/` directory for future reference.
- `train.py` Given an empty model and data loaders, trains that model. It takes the snapshots of the model and ultimately saves the best model (determined by the highest validation F1 score) in the `trained_models/` directory. Along with model paramers, it also saves the running epoch statistics for training and validation loss, F1, accuracy.
- `visualization.py` Plotting of the epoch statistics.
- `predict.py` Uses the given model and the test set images to create the predictions as mask images on the `predictions/` directory. 
- `unet.py` Earlier implementations only left for reference.
- `Deeplab Training.ipynb` Is the training run that generated the model.
- `other_notebooks/` Some of our imporant notebooks that show our progress.
- `other_models/` Some of our imporant models that show our progress.
- `requirements.txt` Some necessary packages that needs to be installed.