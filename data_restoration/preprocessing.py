import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import glob
import h5py
from data_restoration.params import *
from google.cloud import storage

def preprocessed_data(mode='gcloud'):

    '''
    Process l'ensemble des données en batch et les stocke dans un csv
    '''
    # Retrieve `query` data as a DataFrame iterable
    data_processed_path = Path(PATH_PROCESSED_DATA).joinpath("processed_dataset.h5")
    temp_processed_path = Path(PATH_PROCESSED_DATA).joinpath("temp.h5")

    data_processed_exists = data_processed_path.is_file()
    if not data_processed_exists :
        chunk_size = 1000

        filelist = glob.glob(f"{PATH_RAW_DATA}*.jpg")
        chunk_lists = [filelist[i:i+chunk_size] for i in range(0, len(filelist), chunk_size)]

        for chunk_id, chunk in enumerate(chunk_lists) :
            df = pd.DataFrame([load_img(file) for file in chunk], columns=['image'])
            df['heights'] = df['image'].map(lambda x: x.size[1])
            df_filtered = df[df["heights"].between(100,500)]
            df_final = df_filtered["image"].map(lambda x: x.resize((300,300))).map(img_to_array)
            matrix = np.stack(df_final.to_numpy()).astype(np.int16)

        # Save and append the processed chunk to a local CSV at "data_processed_path"
            # df_final.to_csv(
            #     data_processed_path,
            #     mode="w" if chunk_id==0 else "a",
            #     header=False,
            #     index=False,
            #     )


            if chunk_id == 0 :
                with h5py.File(data_processed_path,"w") as f :
                    f.create_dataset("processed_dataset",data=matrix,chunks=matrix.shape,maxshape=(None,300,300,3),compression='gzip',
                    compression_opts=9)
            else :
                with h5py.File(data_processed_path,"a") as f :
                    f["processed_dataset"].resize((f["processed_dataset"].shape[0] + matrix.shape[0]), axis = 0)
                    f["processed_dataset"][-matrix.shape[0]:] = matrix

                if temp_processed_path.is_file() :
                        os.remove(temp_processed_path)

                with h5py.File(temp_processed_path, "w") as temp :
                    temp.create_dataset('_temp_',data=matrix,compression='gzip',compression_opts=9)

            filepath = data_processed_path if chunk_id == 0 else temp_processed_path
            if mode == 'gcloud' :
                client = storage.Client()
                if chunk_id == 0 :
                    try :
                        bucket = client.create_bucket(BUCKET_NAME)
                    except :
                        bucket = client.bucket(BUCKET_NAME)
                else :
                    bucket = client.bucket(BUCKET_NAME)
                blob = bucket.blob(f"dataset/data_restoration_chunk_{chunk_id}")
                blob.upload_from_filename(filepath)
            print('chunk_id : ', chunk_id)

    print('Data ready!')
