import matplotlib.image as mpimg
import numpy as np

from evoline.img.processor import img_float_to_uint8, make_img_overlay, img_crop, IMG_PATCH_SIZE, label_to_img, \
    concatenate_images, value_to_class_value, reflect_image_padding
from evoline.img.visualize import make_img_overlay_gt_pred


def get_prediction_to_img(img, model, padding=0, fill_mode='reflect'):
    data = np.asarray(img_crop(reflect_image_padding(img, padding, fill_mode=fill_mode), IMG_PATCH_SIZE, IMG_PATCH_SIZE, padding))
    # data_node = tf.constant(data)
    # output = tf.nn.softmax(model(data_node))
    output_prediction = model.predict(data)
    output_prediction = np.argmax(output_prediction, axis=1)
    # output_prediction = s.run(output)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction


def get_prediction(img, model, padding=0, fill_mode='reflect'):
    data = np.asarray(img_crop(reflect_image_padding(img, padding, fill_mode=fill_mode), IMG_PATCH_SIZE, IMG_PATCH_SIZE, padding))
    # data_node = tf.constant(data)
    # output = tf.nn.softmax(model(data_node))
    output_prediction = model.predict(data)
    output_prediction = np.argmax(output_prediction, axis=1)
    # output_prediction = s.run(output)

    return output_prediction


def get_prediction_concat(filename, image_idx, model, padding=0):
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction_to_img(img, model, padding)
    cimg = concatenate_images(img, img_prediction)

    return cimg


def get_testing_prediction(filename, image_idx, model, padding=0):
    imageid = "test_%i/test_%i" % (image_idx, image_idx)
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    cimg = get_prediction_to_img(img, model, padding)

    return img_float_to_uint8(cimg)


# Get a concatenation of the prediction and groundtruth for given input file
def get_testing_prediction_overlay(filename, image_idx, model, padding=0):
    imageid = "test_%i/test_%i" % (image_idx, image_idx)
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    cimg = get_prediction_to_img(img, model, padding)
    oimg = make_img_overlay(img, cimg)

    return img_float_to_uint8(oimg)


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, model, padding=0):
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction_to_img(img, model, padding)
    oimg = make_img_overlay(img, img_prediction)

    return oimg


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay_correctness(filename, gt_filename, image_idx, model, padding=0, fill_mode='reflect'):
    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    imageid = "satImage_%.3d" % image_idx
    gt_image_filename = gt_filename + imageid + ".png"
    gt_img = mpimg.imread(gt_image_filename)
    gt_labels = np.array([value_to_class_value(np.mean(patch)) for patch in img_crop(reflect_image_padding(gt_img, padding=padding, fill_mode=fill_mode), IMG_PATCH_SIZE, IMG_PATCH_SIZE, padding)])
    gt_labels_img = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, gt_labels)
    img_prediction = get_prediction_to_img(img, model, padding)
    oimg = make_img_overlay_gt_pred(img, gt_labels_img, img_prediction)

    return oimg
