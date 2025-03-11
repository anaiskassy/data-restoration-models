import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from random import sample

import glob
from PIL import Image

from preprocessing import preprocessed_data

def load_data(size = 1000, randomized = False):

    """
    Loads image data into a DataFrame.

    Parameters:
        size (int or str): Number of files to load ('all' for full list).
        randomized (bool): If True, selects files randomly.

    Returns:
        pd.DataFrame: DataFrame containing image arrays.
    """

    images = np.load("{PATH_PROCESSED_DATA}processed_images.npy")

    if size == "all":
        return images

    size = min(size, len(images))

    if randomized:
        indices = np.random.choice(len(images), size, replace = False)
        return images[indices]

    else:
        return images[:size]
