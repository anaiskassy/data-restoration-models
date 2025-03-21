from data_restoration.evaluation import *
from data_restoration.base_models import *
from data_restoration.load_data import *
from data_restoration.unet_models import *
from data_restoration.unet_models_copy import *
from data_restoration.params import *
from data_restoration.combined_model import *

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


# ------------------- model U-NET copy ----------------------#

if int(MODEL) == 4 :
    # initilization models
    generator = build_reconstruction()
    gen_opti = recon_optimizer_unet()
    discriminator = build_adversarial()
    disc_opti = adv_optimizer_unet()
    print('models initialized')

    # train base model and save weights
    # preprocessing data
    data_train_damaged , expected_output_train = damaging_opti_dataset(data_train)
    print('data preprocessed')

    # reload weights if needed
    if int(RELOAD_W) == 1 :
        local_path_gen = os.path.join(PATH_MODELS,'unet_model','models','generator')
        local_path_dis = os.path.join(PATH_MODELS,'unet_model','models','discriminator')
        l_gen = os.listdir(local_path_gen)
        l_dis = os.listdir(local_path_dis)
        l_dis.sort()
        l_gen.sort()
        path_gen_to_reload = os.path.join(local_path_gen,l_gen[-1])
        path_dis_to_reload = os.path.join(local_path_dis,l_dis[-1])
        generator.load_weights(path_gen_to_reload)
        discriminator.load_weights(path_dis_to_reload)

    # train
    history_gen, history_disc , predictions , progressive_output = train_unet_model_copy(
        data=expected_output_train,
        data_damaged=data_train_damaged,
        generator=generator,
        generator_optimizer=gen_opti,
        discriminator=discriminator,
        discriminator_optimizer=disc_opti,
        epochs=int(N_EPOCHS),
        batch_size=int(BATCH_SIZE),
        chkpt = int(CHECKPOINT), workbook=False
    )
    print('models trained')

    # # evaluation model
    gen_loss, disc_loss, generated_images = evaluation_unet_model(
        data_test=data_test,
        generator=generator,
        discriminator=discriminator,
        workbook=False)
    print('models evaluated')



# ------------------- model combined ----------------------#

if int(MODEL) == 5 :
    # initilization models
    g_model = define_generator()
    c_model = define_critic()
    gan_model = define_gan(generator=g_model,critic=c_model)
    print('models initialized')

    # train base model and save weights
    # preprocessing data
    dataset_damaged , dataset_damaged_input, dataset_expected_output = damaging_opti_dataset_normalized(dataset=data)
    print('data preprocessed')

    # reload weights if needed
    if int(RELOAD_W) == 1 :
        local_path_gen = os.path.join(PATH_MODELS,'combined_model','models','generator')
        local_path_dis = os.path.join(PATH_MODELS,'combined_model','models','discriminator')
        local_path_comb = os.path.join(PATH_MODELS,'combined_model','models','combined')
        l_gen = os.listdir(local_path_gen)
        l_dis = os.listdir(local_path_dis)
        l_comb = os.listdir(local_path_comb)
        l_dis.sort()
        l_gen.sort()
        l_comb.sort()
        path_gen_to_reload = os.path.join(local_path_gen,l_gen[-1])
        path_dis_to_reload = os.path.join(local_path_dis,l_dis[-1])
        path_comb_to_reload = os.path.join(local_path_comb,l_comb[-1])
        g_model.load_weights(path_gen_to_reload)
        c_model.load_weights(path_dis_to_reload)
        gan_model.load_weights(path_comb_to_reload)

    # train

    g_hist, c1_hist , c2_hist, predictions = train(g_model,
                                               c_model,
                                               gan_model,
                                               images_damaged_input=dataset_damaged_input,
                                               expected_outputs=dataset_expected_output,
                                               n_epochs=int(N_EPOCHS),
                                               batch_size=int(BATCH_SIZE),
                                               n_critic=5,chkpt= int(CHECKPOINT), workbook=False)
    print('models trained')
