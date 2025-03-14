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


def evaluation_base_model(data_test,ind_folder = -1) :
    # load data
    # load model

    #return gen_losses, disc_losses , predictions
    pass


def lire_metrics(string):
    return float(string.replace('tf.Tensor(','').replace(', shape=(), dtype=float32)',''))
