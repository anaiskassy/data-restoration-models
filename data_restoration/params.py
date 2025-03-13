import os
import numpy as np

##################  VARIABLES  ##################
PATH_RAW_DATA = os.environ.get("PATH_RAW_DATA")
PATH_RAW_HEAD = os.environ.get("PATH_RAW_HEAD")
PATH_PROCESSED_DATA = os.environ.get("PATH_PROCESSED_DATA")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
PATH_MODELS=os.environ.get("PATH_MODELS")

N_EPOCHS=os.environ.get("N_EPOCHS")
CHECKPOINT=os.environ.get("CHECKPOINT")
RELOAD_W=os.environ.get("RELOAD_W")
BATCH_SIZE=os.environ.get("BATCH_SIZE")
