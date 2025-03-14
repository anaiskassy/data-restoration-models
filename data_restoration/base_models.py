import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers,Sequential
import time
import h5py
import os
from data_restoration.params import *

def init_base_model(lr = 1e-4):
    generator = make_generator_base_model()
    discriminator = make_discriminator_base_model()
    gen_opti = generator_optimizer_base(lr)
    disc_opti = discriminator_optimizer_base(lr)
    return generator,gen_opti,discriminator,disc_opti

def init_model_2(lr = 1e-4):
    generator = make_generator_model_2()
    discriminator = make_discriminator_model_2()
    gen_opti = generator_optimizer_base(lr)
    disc_opti = discriminator_optimizer_base(lr)
    return generator,gen_opti,discriminator,disc_opti


def make_generator_base_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False,input_shape=(64, 64, 3)))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='sigmoid'))

    return model

def make_generator_model_2():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False,input_shape=(64, 64, 3)))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))

    return model

def generator_loss_base(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_optimizer_base(learning_rate = 1e-4):
    return tf.keras.optimizers.legacy.Adam(learning_rate)


def make_discriminator_base_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=(64, 64, 3)))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(3, 3), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

def make_discriminator_model_2():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=(16, 16, 3)))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(3, 3), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

def discriminator_loss_base(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def discriminator_optimizer_base(learning_rate = 1e-4):
    return tf.keras.optimizers.legacy.Adam(learning_rate)

@tf.function
def train_step_base_model(images,images_damaged,
                          generator,generator_optimizer,
                          discriminator,discriminator_optimizer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(images_damaged, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss_base(fake_output)
      disc_loss = discriminator_loss_base(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train_base_model(data,data_damaged,
          generator,generator_optimizer,
          discriminator,discriminator_optimizer,
          epochs,batch_size = 100,
          chkpt = 1, workbook=True):

    progressive_output = []
    history_gen = []
    history_disc = []
    start = time.time()
    nb_batches = int(data.shape[0]/batch_size) + 1

    for epoch in range(epochs):
        for i in range(nb_batches) :
            image_batch = data[batch_size*i : (i+1)*batch_size,:,:,:]
            image_damaged_batch = data_damaged[batch_size*i : (i+1)*batch_size,:,:,:]
            loss_gen, loss_disc = train_step_base_model(image_batch,image_damaged_batch,
                                                        generator=generator,
                                                        generator_optimizer=generator_optimizer,
                                                        discriminator=discriminator,
                                                        discriminator_optimizer=discriminator_optimizer)
            history_disc.append(loss_disc)
            history_gen.append(loss_gen)
            print(loss_gen)

            print('epoch', epoch,'batch',i,'/', nb_batches, time.time()-start, 'loss_gen', loss_gen, 'loss_disc', loss_disc)

        if (epoch + 1)%chkpt == 0 or epoch == epochs - 1:
            # Show output pour faire une GIF:
            prediction = generator(np.expand_dims(data_damaged[0],axis=0), training=False)[0,:,:,:] # training = False ne modifie pas les poids des couches de neuronnes
            progressive_output.append(prediction)

            # save metrics
            metrics = pd.DataFrame({'history_generator_loss' : history_gen,'history_discriminator_loss' : history_disc})
            local_path_metrics = os.path.join(PATH_MODELS,'base_model','metrics', f"{time.strftime('%Y%m%d-%H%M%S')}-epoch{epoch+1}.csv")
            path_dir_metrics = os.path.join(PATH_MODELS,'base_model','metrics')
            if workbook :
                local_path_metrics = os.path.join('..',local_path_metrics)
                path_dir_metrics = os.path.join('..',path_dir_metrics)
            if not os.path.exists(path_dir_metrics) :
                os.makedirs(path_dir_metrics)
            metrics.to_csv(local_path_metrics)
            print('metrics saved')

            # save le model
            local_path_gen = os.path.join(PATH_MODELS,'base_model','models','generator')
            local_path_dis = os.path.join(PATH_MODELS,'base_model','models','discriminator')

            if workbook :
                local_path_gen = os.path.join('..',local_path_gen)
                local_path_dis = os.path.join('..',local_path_dis)

            path_gen = os.path.join(local_path_gen,f"{time.strftime('%Y%m%d-%H%M%S')}-epoch{epoch+1}.h5")
            path_disc =os.path.join(local_path_dis,f"{time.strftime('%Y%m%d-%H%M%S')}-epoch{epoch+1}.h5")

            if not os.path.exists(local_path_gen) :
                os.makedirs(local_path_gen)
            if not os.path.exists(local_path_dis) :
                os.makedirs(local_path_dis)
            generator.save(path_gen)
            discriminator.save(path_disc)
            print('models saved')


    # Generate after the final epoch
    predictions = generator(data_damaged[:20], training=False)
    plt.imshow(predictions[0]);

    return history_gen, history_disc , predictions , progressive_output
