"""

Utils for adversarial training

NOTE:
 - At this stage it only works for ResNet50

Author: Simon Thomas
Date: 29-Sep-2020

"""

import numpy as np

# Preprocessing Tools
MEANS = np.array([103.939, 116.779, 123.68])

def preprocess(img):
    """
    Preprocesses an image(s) [0,255] by
    substracting the mean channel value from the ResNet50
    :param img: the image(s) to preprocess
    :return: img - resnet_means
    """
    if len(img.shape) != 4:
        img = np.expand_dims(img, 0)
    return img - MEANS

def deprocess(img):
    """
    Deprocesses an image(s) [-127, 127] by adding
    the mean channel value from ResNet50
    :param img: img of type float
    :return: img of type uint8
    """
    img = img + MEANS
    return np.clip(img, 0, 255).astype("uint8")
