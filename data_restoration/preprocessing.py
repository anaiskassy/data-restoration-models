import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import glob
from data_restoration.params import *

def preprocessed_data():

    '''
    Process l'ensemble des données en batch et les stocke dans un csv
    '''
    # Retrieve `query` data as a DataFrame iterable
    data_processed_path = Path(PATH_PROCESSED_DATA).joinpath("processed_images.csv")

    data_processed_exists = data_processed_path.is_file()
    if not data_processed_exists :
        chunk_size = 1000

        filelist = glob.glob(f"{PATH_RAW_DATA}*.jpg")
        chunk_lists = [filelist[i:i+chunk_size] for i in range(0, len(filelist), chunk_size)]

        for chunk_id, chunk in enumerate(chunk_lists) :
            df = pd.DataFrame([load_img(file) for file in chunk], columns=['image'])
            df.loc['heights'] = df['image'].map(lambda x: x.size[1])
            df_filtered = df[df["heights"].between(100,500)]
            df_filtered["image"] = df_filtered["image"].map(lambda x: x.resize((300,300))).map(img_to_array)/255
            df_final = df_filtered[["image"]]

        # Save and append the processed chunk to a local CSV at "data_processed_path"
            df_final.to_csv(
                data_processed_path,
                mode="w" if chunk_id==0 else "a",
                header=False,
                index=False,
                )
            print(chunk_id)
    print('Data ready!')
