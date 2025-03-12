import pandas as pd
import numpy as np
from pathlib import Path
from random import sample
from data_restoration.params import *
from data_restoration.preprocessing import preprocessed_data
import h5py
from google.cloud import storage


def load_data_from_local(nrows='all',workbook=False,chunk_id=None):
    """
    load data h5 from local
    """
    if chunk_id == None :
        if workbook :
            data_processed_path = Path(f'.{PATH_PROCESSED_DATA}').joinpath("processed_dataset.h5")
        else :
            data_processed_path = Path(PATH_PROCESSED_DATA).joinpath("processed_dataset.h5")
    else :
        if workbook :
            data_processed_path = Path(f'.{PATH_PROCESSED_DATA}').joinpath(f"processed_dataset_chunk_{chunk_id}.h5")
        else :
            data_processed_path = Path(PATH_PROCESSED_DATA).joinpath(f"processed_dataset_chunk_{chunk_id}.h5")
    if not data_processed_path.is_file() :
        print('No data at', data_processed_path)
        #preprocessed_data()

    with h5py.File(data_processed_path,"r") as dset :
        if nrows == 'all' :
            images = dset['processed_dataset'][:]
        else :
            images = dset['processed_dataset'][:nrows]
    return images


def load_data_from_cloud(n_chunk=1,workbook=False):
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="dataset"))
    for n in range(n_chunk) :
        if workbook :
            data_processed_path = Path(f'.{PATH_PROCESSED_DATA}').joinpath(f"processed_dataset_chunk_{n}.h5")
        else :
            data_processed_path = Path(PATH_PROCESSED_DATA).joinpath(f"processed_dataset_chunk_{n}.h5")
        blob_n = blobs[n]
        blob_n.download_to_filename(data_processed_path)
        with h5py.File(data_processed_path,"r") as dset :
            images_chunk = dset['processed_dataset'][:]
        images = images_chunk if n == 0 else np.vstack([images,images_chunk])

    return images
