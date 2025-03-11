import pandas as pd
import numpy as np
from pathlib import Path
from random import sample
from data_restoration.params import *
from data_restoration.preprocessing import preprocessed_data


def load_data(nrows = 1000,chunksize=None):
    """
    Loads image data into a DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing image arrays.
    """
    data_processed_path = Path(PATH_PROCESSED_DATA).joinpath("processed_images.csv")
    if not data_processed_path.is_file() :
        preprocessed_data()

    if nrows == 'all' :
        if chunksize == None :
            images = pd.read_csv(data_processed_path)
        else :
            images = pd.read_csv(data_processed_path,chunksize=chunksize)
    else :
        if chunksize == None :
            images = pd.read_csv(data_processed_path,nrows=nrows)
        else :
            images = pd.read_csv(data_processed_path,chunksize=chunksize,nrows=nrows)
    return images
