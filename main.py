from data_restoration.evaluation import *
from data_restoration.base_models import *
from data_restoration.load_data import *
from data_restoration.unet_models import *
from data_restoration.params import *

# ------------------- load data ----------------------#

n_rows = N_ROWS if N_ROWS == 'all' else int(N_ROWS)

if DATASET == 'cat_heads' :
    data = load_data_head_small(nrows=n_rows,mode=MODE)
    print('dataset used = processed_dataset_head_small.h5')
if DATASET == 'cats' :
    data = load_data_small(nrows=n_rows,mode=MODE)
    print('dataset used = processed_dataset_small.h5')
if DATASET == 'both' :
    data = load_data_small_all(nrows=n_rows,mode=MODE)
    print('dataset used = processed_dataset_head_small.h5 and processed_dataset_small.h5')

data_train = data[:-100]
data_test = data[-100:]
print(f'train = {data_train.shape[0]} lines --- test = 100 lines')

# ------------------- base model 1 ----------------------#

if int(MODEL) == 1 :
    # initilization models
    generator,gen_opti,discriminator,disc_opti = init_base_model()
    print('models initialized')

    # train base model and save weights
    history_gen, history_disc , predictions_finales , progressive_output = run_base_model(
        data_train=data_train,
        generator=generator,
        gen_opti=gen_opti,
        discriminator=discriminator,
        disc_opti=disc_opti,
        n_epochs=int(N_EPOCHS),
        batch_size=int(BATCH_SIZE),
        reload_w=int(RELOAD_W),
        checkpoint=int(CHECKPOINT),
        workbook=False
    )
    print('models trained')

    # evaluation model
    gen_loss, disc_loss, generated_images = evaluation_base_model(
        data_test=data_test,
        generator=generator,
        discriminator=discriminator,
        workbook=False)
    print('models evaluated')


# ------------------- base model 2 ----------------------#

if int(MODEL) == 2 :
    # initilization models
    generator,gen_opti,discriminator,disc_opti = init_model_2()
    print('models initialized')

    # train base model and save weights
    history_gen, history_disc , predictions_finales , progressive_output = run_base_model_2(
        data_train=data_train,
        generator=generator,
        gen_opti=gen_opti,
        discriminator=discriminator,
        disc_opti=disc_opti,
        n_epochs=int(N_EPOCHS),
        batch_size=int(BATCH_SIZE),
        reload_w=int(RELOAD_W),
        checkpoint=int(CHECKPOINT),
        workbook=False
    )
    print('models trained')

    # # evaluation model
    gen_loss, disc_loss, generated_images = evaluation_base_model_2(
        data_test=data_test,
        generator=generator,
        discriminator=discriminator,
        workbook=False)
    print('models evaluated')


# ------------------- model U-NET ----------------------#

if int(MODEL) == 3 :
    # initilization models
    generator,gen_opti,discriminator,disc_opti = init_unet_model()
    print('models initialized')

    # train base model and save weights
    history_gen, history_disc , predictions_finales , progressive_output = run_unet_model(
        data_train=data_train,
        generator=generator,
        gen_opti=gen_opti,
        discriminator=discriminator,
        disc_opti=disc_opti,
        n_epochs=int(N_EPOCHS),
        batch_size=int(BATCH_SIZE),
        reload_w=int(RELOAD_W),
        checkpoint=int(CHECKPOINT),
        workbook=False
    )
    print('models trained')

    # # evaluation model
    gen_loss, disc_loss, generated_images = evaluation_unet_model(
        data_test=data_test,
        generator=generator,
        discriminator=discriminator,
        workbook=False)
    print('models evaluated')
