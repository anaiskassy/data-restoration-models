import os
import numpy as np

##################  LOCAL PATH  ##################
PATH_RAW_DATA = os.environ.get("PATH_RAW_DATA")
PATH_RAW_HEAD = os.environ.get("PATH_RAW_HEAD")
PATH_PROCESSED_DATA = os.environ.get("PATH_PROCESSED_DATA")
PATH_MODELS=os.environ.get("PATH_MODELS")

##################  VARIABLES  ##################
N_EPOCHS=os.environ.get("N_EPOCHS")
CHECKPOINT=os.environ.get("CHECKPOINT")
RELOAD_W=os.environ.get("RELOAD_W")
BATCH_SIZE=os.environ.get("BATCH_SIZE")
MODE=os.environ.get("MODE")
MODEL=os.environ.get("MODEL")
DATASET=os.environ.get("DATASET")
N_ROWS=os.environ.get("N_ROWS")

##################  CLOUD STORAGE  ##################
BUCKET_NAME = os.environ.get("BUCKET_NAME")
BUCKET_NAME_SMALL = os.environ.get("BUCKET_NAME_SMALL")
