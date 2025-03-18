import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img
import io
from api.utils import *

app = FastAPI()

model1 = load_model(number=1)
model2 = load_model(number=2)
model3 = load_model(number=3)
model4 = load_model(number=4)


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/preproc")
async def preproc(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_array = img_to_array(image.resize((64,64)))

    np.save('temp.npy',image_array)

    image_processed , image_damaged = preprocessing_image(image_array, 2)
    image_damaged_img = array_to_img(image_damaged)

    image_damaged_byt = io.BytesIO()
    image_damaged_img.save(image_damaged_byt,format='PNG')
    image_damaged_byt.seek(0)
    resp_im_dam = image_damaged_byt.getvalue()
    return Response(content = resp_im_dam, media_type="image/png")


@app.get("/predict")
def predict(model:int):
    image_array = np.load('temp.npy').astype(int)
    model_nb = int(model)
    model = load_model(model_nb)

    image_processed , image_damaged = preprocessing_image(image_array, model_nb)
    prediction = model.predict(np.expand_dims(image_processed,axis=0))[0]
    image_damaged , image_rebuild = postprocessing_image(image_array, image_damaged, prediction, model_nb)
    image_rebuild_img = array_to_img(image_rebuild)

    image_rebuild_byt = io.BytesIO()
    image_rebuild_img.save(image_rebuild_byt,format='PNG')
    image_rebuild_byt.seek(0)
    resp_im_reb = image_rebuild_byt.getvalue()
    return Response(content = resp_im_reb, media_type="image/png")


@app.get("/")
def root():
    return {'I am watching you': 'Do not turn around'}
