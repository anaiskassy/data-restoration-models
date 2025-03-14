from data_restoration.evaluation import *
from data_restoration.base_models import *
from data_restoration.load_data import *

if int(BASE_MODEL) == 1 :
    # load head cats
    data = load_data_head_small(nrows='all',mode='gcloud')
    data_train = data[:-100]
    data_test = data[-100:]
    print('dataset used = processed_dataset_head_small.h5')
    print('train = dataset[:-100] --- test = dataset[-100:]')

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

if int(BASE_MODEL) == 2 :
    # load head cats
    data = load_data_head_small(nrows='all',mode='gcloud')
    data_train = data[:-100]
    data_test = data[-100:]
    print('dataset used = processed_dataset_head_small.h5')
    print('train = dataset[:-100] --- test = dataset[-100:]')

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
