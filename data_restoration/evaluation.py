from data_restoration.damaging import *
from data_restoration.base_models import *
from data_restoration.params import *
import pandas as pd
import os
import time


def run_base_model(data_train,
                   generator,gen_opti,
                   discriminator,disc_opti,
                   n_epochs=5,batch_size=100,workbook=False,
                   checkpoint=1,reload_w=0) :
    # preprocessing data
    data_train_damaged = damaging_dataset(data_train) /255
    data_train = data_train / 255
    print('data preprocessed')

    # reload weights if needed
    if reload_w == 1 :
        local_path_gen = os.path.join(PATH_MODELS,'base_model','models','generator')
        local_path_dis = os.path.join(PATH_MODELS,'base_model','models','discriminator')
        if workbook :
            local_path_gen = os.path.join('..',local_path_gen)
            local_path_dis = os.path.join('..',local_path_dis)
        l_gen = os.listdir(local_path_gen)
        l_dis = os.listdir(local_path_dis)
        l_dis.sort()
        l_gen.sort()
        path_gen_to_reload = os.path.join(local_path_gen,l_gen[-1])
        path_dis_to_reload = os.path.join(local_path_dis,l_dis[-1])

        generator.load_weights(path_gen_to_reload)
        discriminator.load_weights(path_dis_to_reload)

    # train
    history_gen, history_disc , predictions , progressive_output = train_base_model(
        data=data_train,
        data_damaged=data_train_damaged,
        generator=generator,
        generator_optimizer=gen_opti,
        discriminator=discriminator,
        discriminator_optimizer=disc_opti,
        epochs=n_epochs,
        batch_size=batch_size,
        chkpt = checkpoint, workbook=workbook
    )

    return history_gen, history_disc , predictions , progressive_output


def evaluation_base_model(data_test,generator,discriminator,workbook=False) :
    # preprocessing
    data_test_damaged = damaging_dataset(data_test) /255
    data_test = data_test / 255

    # testing models
    generated_images = generator(data_test_damaged, training=False)
    real_output = discriminator(data_test, training=False)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss_base(fake_output)
    disc_loss = discriminator_loss_base(real_output, fake_output)

    metrics = pd.DataFrame({'history_generator_loss' : gen_loss,'history_discriminator_loss' : disc_loss})
    local_path_metrics = os.path.join(PATH_MODELS,'base_model','metrics_test', f"{time.strftime('%Y%m%d-%H%M%S')}.csv")
    path_dir_metrics = os.path.join(PATH_MODELS,'base_model','metrics_test')
    if workbook :
        local_path_metrics = os.path.join('..',local_path_metrics)
        path_dir_metrics = os.path.join('..',path_dir_metrics)
    if not os.path.exists(path_dir_metrics) :
        os.makedirs(path_dir_metrics)
    metrics.to_csv(local_path_metrics)
    print('metrics saved')

    return gen_loss, disc_loss, generated_images


def lire_metrics(string):
    return float(string.replace('tf.Tensor(','').replace(', shape=(), dtype=float32)',''))
