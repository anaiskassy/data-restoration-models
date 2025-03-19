from pathlib import Path
from google.cloud import storage
from data_restoration.params import *
from data_restoration.base_models import make_generator_base_model, make_generator_model_2
from data_restoration.unet_models import make_generator_unet_model
from data_restoration.combined_model import define_generator
from data_restoration.damaging import damaging,damaging_opti,damaging_opti_normalized, postprocessing_dataset_normalized, postprocessing_dataset


# Loading models

def load_model(number=1) :
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME_SMALL)
    if number == 1 :
        path_model1 = Path("models").joinpath("generator-base-1.h5")
        if not path_model1.is_file() :
            blob_1 = bucket.blob("model-base/generator/20250317-001957-gen-epoch75.h5")
            blob_1.download_to_filename(path_model1)
        model = make_generator_base_model()
        model.load_weights(path_model1)

    if number == 2 :
        path_model2 = Path("models").joinpath("generator-base-2.h5")
        if not path_model2.is_file() :
            blob_2 = bucket.blob(f"model-base-2/generator/20250317-054006-gen-epoch80.h5")
            blob_2.download_to_filename(path_model2)
        model = make_generator_model_2()
        model.load_weights(path_model2)

    if number == 3 :
        path_model3 = Path("models").joinpath("generator-unet.h5")
        if not path_model3.is_file() :
            blob_3 = bucket.blob(f"model-unet/generator/20250316-101534-gen-epoch60.h5")
            blob_3.download_to_filename(path_model3)
        model = make_generator_unet_model()
        model.load_weights(path_model3)

    if number == 4 :
        path_model4 = Path("models").joinpath("generator-combined.h5")
        if not path_model4.is_file() :
            blob_4 = bucket.blob(f"model-combined/generator/generator-VF.h5")
            blob_4.download_to_filename(path_model4)
        model = define_generator()
        model.load_weights(path_model4)
    return model


def preprocessing_image(image, model_nb=1) :
    if model_nb == 1 :
        image_processed = damaging(X=image,percent=5,random=False)
        image_processed = image_processed / 255
        image_damaged = image_processed.copy()

    if model_nb == 2 or model_nb == 3:
        image_processed , expected  = damaging_opti(X=image,n_dim=16)
        image_processed = image_processed / 255
        image_damaged = image_processed.copy()

    if model_nb == 4 :
        image_damaged, image_processed, expected = damaging_opti_normalized(X=image,n_dim=16)

    return image_processed , image_damaged


def postprocessing_image(image, image_damaged, prediction, model_nb=1) :
    if model_nb == 1 :
        image_rebuild = prediction

    if model_nb == 2 or model_nb == 3 :
        dataset_pred = np.expand_dims(prediction,axis=0)
        dataset_im = np.expand_dims(image,axis=0)
        image_rebuild = postprocessing_dataset(dataset_im,dataset_pred,n_dim=16)[0]

    if model_nb == 4 :
        dataset_pred = np.expand_dims(prediction,axis=0)
        dataset_im = np.expand_dims(image,axis=0)
        dataset_dam = np.expand_dims(image_damaged,axis=0)
        dataset_damaged, dataset_rebuild = postprocessing_dataset_normalized(
            dataset=dataset_im,
            dataset_damaged=dataset_dam,
            dataset_generated=dataset_pred,
            n_dim=16)

        image_damaged = dataset_damaged[0]
        image_rebuild = dataset_rebuild[0]

    return image_damaged , image_rebuild


def model_selection(model1,model2,model3,model4,model_nb):
    if model_nb == 1 :
        return model1
    if model_nb == 2 :
        return model2
    if model_nb == 3 :
        return model3
    if model_nb == 4 :
        return model4
