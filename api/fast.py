import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from data_restoration.preprocessing import preprocessed_data_small
from keras.preprocessing.image import img_to_array
import io

app = FastAPI()

def preprocess_image(image_bytes):
    loaded_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    loaded_img = loaded_img.resize((64, 64))
    loaded_img_array = img_to_array(loaded_img) / 255
    return np.expand_dims(loaded_img_array, axis=0)

def postprocess_image(image_array):
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image_io = io.BytesIO()
    image.save(image_io, format="PNG")
    image_io.seek(0)
    return image_io

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = load_model()

@app.post("/predict")
def predict(file: UploadFile = File(...)):

    image_bytes = file.file.read()
    image = preprocess_image(image_bytes)

    completed_img = model.predict(image)[0]

    image_io = postprocess_image(completed_img)

    return Response(content=image_io.getvalue(), media_type="image/png")


@app.get("/")
def root():
    return {'I am watching you': 'Do not turn around'}
