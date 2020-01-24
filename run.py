from dataset import get_dataloaders
from train import train_model
from hyper import tune_hyper
from visualization import plot_epoch_stats
from predict import write_predictions
from mask_to_submission import create_submission

from torchvision.models.segmentation import deeplabv3_resnet101


def main():
    print('Loading training and validation set images...')
    
    # Getting training and validation data
    dataloaders = get_dataloaders(
        random_transform=True, random_rotate=True,
        random_hflip=True, random_vflip=True,
        train_ratio=0.8
    )
    
    print('Starting training of model...')

    # # The way the winning submission was trained, takes about ~3 hours with GPU
    # model = deeplabv3_resnet101(num_classes=1, pretrained=False)
    # model, epoch_stats, model_path, best_f1 = train_model(
    #     model, dataloaders, num_epochs=5, learning_rate=0.0002, deeplab=True)

    print('Loading best model and writing predictions for test set...')

    # Getting the best model from training and using it on test set
    empty_model = deeplabv3_resnet101(num_classes=1, pretrained=False)
    model_path = 'trained_models/20191209-020809-net.pth' # submission: 31396
    write_predictions(empty_model, model_path, deeplab=True, patched=False, image_path='predictions')

    print('Creating csv submission from prediction images...')
    
    # Creating csv
    create_submission()
    
    print('Done.')
    
    
if __name__ == "__main__":
    main()