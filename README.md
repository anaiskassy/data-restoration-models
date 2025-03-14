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
