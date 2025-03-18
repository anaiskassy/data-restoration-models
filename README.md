To load data :
! curl -L -o ./raw_data/cats.zip\
  https://www.kaggle.com/api/v1/datasets/download/denispotapov/cat-breeds-dataset-cleared
! curl -L -o ./raw_data/cats_heads.zip\
  https://www.kaggle.com/datasets/borhanitrash/cat-dataset

to do for the architecture :
  - data folder
    - raw_data as subfolder to load data from Kaggle (images + cvs)
    - preprocessed_data as subfolder
  - notebooks folder

to train and test models :
1. Updating .env variables
  - N_EPOCHS : int
  - CHECKPOINT : nombre d'epochs avant une sauvegarde des modèles et metrics
  - RELOAD_W : values ['1' : reload les dernier poid du modèle,'0' : initialise un modèle]
  - BATCH_SIZE : int
  - MODE: values ['local','gcloud'] pour charger les données
  - MODEL : values ['1' : base model avec input et output de 64x64,
                    '2' : base model avec input 64x64 et output 16x16,
                    '3' : modèle UNET
                    '4' : modèle UNET 2
                    '5' : modèle combined]]
  - DATASET : values ['cat_heads','cats','both']
  - N_ROWS : values [int ou 'all'] nombre d'images à charger par dataset

2. Launching main.py



to launch API :
> docker build \
  --platform linux/amd64 \
  -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/data-restoration-models/$GAR_IMAGE:prod .

> docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/data-restoration-models/$GAR_IMAGE:prod

> gcloud run deploy --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/data-restoration-models/$GAR_IMAGE:prod --memory $GAR_MEMORY --region $GCP_REGION --env-vars-file .env.yaml
