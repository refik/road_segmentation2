import glob
import re

from keras import Input, Model
from keras.layers import Conv2D, UpSampling2D, concatenate, Dropout, MaxPooling2D, Dense, Flatten


def get_unet(pretrained_weights=None, input_shape=(16, 16, 3)):
    inputs = Input(input_shape)

    conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up6 = Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv4))
    merge6 = concatenate([conv3, up6], axis=3)
    conv6 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv2, up7], axis=3)
    conv7 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv1, up8], axis=3)
    conv8 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    conv10 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(conv8)

    model = Model(input=inputs, output=conv10)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def save_model(model: Model, name, directory="pretrained_cnn/models"):
    new_filename = f"{directory}/{name}_0.hdf5"
    files = glob.glob(f"{directory}/{name}*.hdf5")
    if new_filename in files:
        reg_num = re.compile("\d+")
        last_num = int([reg_num.findall(file)[0] for file in files if reg_num.match(file)][-1])
        next_num = last_num + 1
        new_filename = f"{directory}/{name}_{next_num}.hdf5"

    model.save(new_filename)
