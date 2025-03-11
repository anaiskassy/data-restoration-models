import pandas as pd
import numpy as np
from pathlib import Path
from random import sample
from data_restoration.params import *
from data_restoration.preprocessing import preprocessed_data
import h5py
from google.cloud import storage


def load_data(nrows = 1000,workbook = False):
    """
    Loads image data into a DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing image arrays.
    """

    if workbook :
        data_processed_path = Path(f'.{PATH_PROCESSED_DATA}').joinpath("processed_dataset.h5")
    else :
        data_processed_path = Path(PATH_PROCESSED_DATA).joinpath("processed_dataset.h5")
    if not data_processed_path.is_file() :
        print(data_processed_path)
        print('no data')
        preprocessed_data()

    if nrows == 'all' :
        with h5py.File(data_processed_path,"r") as f :
            images = f['processed_dataset'][:]
        # if chunksize == None :
        #     images = pd.read_csv(data_processed_path,header=None)
        # else :
        #     images = pd.read_csv(data_processed_path,chunksize=chunksize,header=None)
    else :
        with h5py.File(data_processed_path,"r") as f :
            images = f['processed_dataset'][:nrows]
        # if chunksize == None :
        #     images = pd.read_csv(data_processed_path,nrows=nrows,header=None)
        # else :
        #     images = pd.read_csv(data_processed_path,chunksize=chunksize,nrows=nrows,header=None)
    return images
