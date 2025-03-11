import pandas as pd
import numpy as np

import tensorflow
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import glob

from params import PATH_PROCESSED_DATA, PATH_RAW_DATA

def preprocessed_data():

    '''
    prend une image en array de taille random et retourne une image en array 300 x 300
    '''
    filelist = glob.glob(f"{PATH_RAW_DATA}/*/*.jpg")
    print(filelist)

    df = pd.DataFrame([load_img(file) for file in filelist], columns=['image'])

    df['heights'] = df['image'].map(lambda x: x.size[1])
    df_filtered = df[df["heights"].between(100,500)]

    df_filtered.reset_index(drop = True, inplace = True)

    df_filtered["image"] = df_filtered["image"].map(lambda x: x.resize((300,300))).map(img_to_array)/255

    df_final = df_filtered[["image"]]

    return df_final

def save_data(df_final):

    np.save(f"{PATH_PROCESSED_DATA}/processed_images.npy", df_final["image"].values)
    df_final.to_csv(f"{PATH_PROCESSED_DATA}/processed_images.csv", index = False)
