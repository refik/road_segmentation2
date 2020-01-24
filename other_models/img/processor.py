import os

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

IMG_PATCH_SIZE = 16
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Extract label images
def extract_labels(filename, num_images, verbose=False):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    all_loaded = True
    for i in num_images:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if verbose:
                print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        elif verbose:
            print('File ' + image_filename + ' does not exist')
            all_loaded = False

    if all_loaded:
        print("Finish loading all images")

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, 0) for i in
                  range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def extract_labels_base(filename, num_images, width, height, expand_3d=True, verbose=False):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    all_loaded = True
    for i in num_images:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if verbose:
                print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        elif verbose:
            print('File ' + image_filename + ' does not exist')
            all_loaded = False

    if all_loaded:
        print("Finish loading all images")

    num_images = len(gt_imgs)
    gt_patches = [img_crop_base(gt_imgs[i], width, height) for i in
                  range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    if expand_3d:
        data = np.expand_dims(data, axis=3)

    return data

# Extract patches from a given image
def img_crop(im, w, h, padding=0):
    list_patches = []
    imgwidth = im.shape[1]
    imgheight = im.shape[0]
    is_2d = len(im.shape) < 3
    for i in range(padding, imgheight - padding, h):
        for j in range(padding, imgwidth - padding, w):
            if is_2d:
                im_patch = im[i - padding: i + h + padding,
                           j - padding: j + w + padding]
            else:
                im_patch = im[i - padding: i + h + padding,
                           j - padding: j + w + padding, :]

            list_patches.append(im_patch)
    return list_patches


def img_crop_base(im, w, h):
    list_patches = []
    imgwidth = im.shape[1]
    imgheight = im.shape[0]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[i: i + w,
                           j: j + h]
            else:
                im_patch = im[i: i + w,
                           j: j + h,
                           :]

            list_patches.append(im_patch)
    return list_patches


# padding = 0 is equivalent to img_crop function
def create_patches_with_padding(img, padding=0, patch_size=(16, 16)):
    list_patches = []
    imgwidth = img.shape[1]
    imgheight = img.shape[0]
    is_2d = len(img.shape) < 3
    for i in range(padding, imgheight + padding, patch_size[0] + 2 * padding):
        for j in range(padding, imgwidth + padding, patch_size[1] + 2 * padding):
            if is_2d:
                list_patches.append(img[i - padding: i + patch_size[0] + padding,
                                    j - padding: j + patch_size[1] + padding])
            else:
                list_patches.append(img[i - padding: i + patch_size[0] + padding,
                                    j - padding: j + patch_size[1] + padding, :])
    return list_patches


def read_img_file(dir, img_id, verbose=False):
    imageid = "satImage_%.3d" % img_id
    image_filename = dir + imageid + ".png"

    if os.path.isfile(image_filename):
        if verbose:
            print('Loading ' + image_filename)
        return mpimg.imread(image_filename)
    elif verbose:
        all_loaded = False
        print('File ' + image_filename + ' does not exist')

    return None


def reflect_image_padding(img, padding=0, fill_mode='reflect'):
    if fill_mode:
        padding = (padding, padding)
        if len(img.shape) < 3:
            return np.lib.pad(img, (padding, padding), fill_mode)
        else:
            return np.lib.pad(img, (padding, padding, (0, 0)), fill_mode)
    else:
        return img


def extract_data(filename, num_images, padding=0, fill_mode='reflect', verbose=False):
    # """Extract the images into a 4D tensor [image index, y, x, channels].
    # Values are rescaled from [0, 255] down to [-0.5, 0.5].
    # """
    imgs = []
    all_loaded = True
    for i in num_images:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if verbose:
                print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        elif verbose:
            all_loaded = False
            print('File ' + image_filename + ' does not exist')

    if all_loaded:
        print("Finish loading all images")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(reflect_image_padding(imgs[i], padding=padding, fill_mode=fill_mode), IMG_PATCH_SIZE, IMG_PATCH_SIZE,
                 padding)
        for i in
        range(num_images)]

    data = [img_patches[i][j] for i in range(len(img_patches)) for j in
            range(len(img_patches[i]))]

    return np.asarray(data)


def extract_data_base(filename, num_images, width, height, verbose=False):
    # """Extract the images into a 4D tensor [image index, y, x, channels].
    # Values are rescaled from [0, 255] down to [-0.5, 0.5].
    # """
    imgs = []
    all_loaded = True
    for i in num_images:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            if verbose:
                print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        elif verbose:
            all_loaded = False
            print('File ' + image_filename + ' does not exist')

    if all_loaded:
        print("Finish loading all images")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop_base(imgs[i], width, height)
        for i in
        range(num_images)]

    data = [img_patches[i][j] for i in range(len(img_patches)) for j in
            range(len(img_patches[i]))]

    return np.asarray(data)


# Assign a label to a patch v
# one hot encode
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


def value_to_class_value(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return 1
    else:  # bgrd
        return 0
