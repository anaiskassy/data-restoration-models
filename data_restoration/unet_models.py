import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers,Sequential
from keras.losses import MeanSquaredError , BinaryCrossentropy
import time
import h5py
import os
from data_restoration.params import *


def wasserstein_loss(y_true,y_pred):
    #y_true = 1 si vrai / -1 si faux
    return tf.reduce_mean(y_true*y_pred)


def init_unet_model():
    generator_unet = make_generator_unet_model()
    discriminator_unet = make_discriminator_unet_model()
    gen_opti_unet = generator_optimizer_unet()
    disc_opti_unet = discriminator_optimizer_unet()
    return generator_unet,gen_opti_unet,discriminator_unet,disc_opti_unet

def make_generator_unet_model() :
    model = tf.keras.Sequential()
    # normalization
    model.add(layers.Input(shape=(64,64,3)))
    model.add(layers.Normalization())
    #---------------encoding---------------#
    # down 1
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 2
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 3
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 4
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 5
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 6
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    #---------------decoding---------------#
    # up 6
    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # up 5
    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # up 4
    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # up 3
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # up 2
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # up 1
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # out
    model.add(layers.Conv2DTranspose(3, (5, 5), activation='softmax', strides=(1, 1), padding='same', use_bias=False))
    return model

def generator_optimizer_unet():
    return tf.keras.optimizers.legacy.Adam(learning_rate=.0002,beta_1=.5,clipnorm=1.)



def generator_loss_unet(fake_output):
    #cross_entropy = BinaryCrossentropy()
    # loss_gan = cross_entropy(tf.ones_like(fake_output), fake_output)
    # --------------#
    # test avec Wasserstein loss
    loss_gan = wasserstein_loss(tf.ones_like(fake_output), fake_output)
    return loss_gan

# def pixel_loss_unet(generated_image,expected_image):
#     mse = MeanSquaredError()
#     loss_pixel = tf.sqrt(mse(expected_image,generated_image)) * 1e-4
#     return loss_pixel


def make_discriminator_unet_model() :
    model = tf.keras.Sequential()
    # normalization
    model.add(layers.Input(shape=(16,16,3)))
    model.add(layers.Normalization())
    # encoding
    # down 1
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 2
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 3
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # down 4
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # out
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model


def discriminator_optimizer_unet() :
    return tf.keras.optimizers.legacy.Adam(learning_rate=.0002,beta_1=.5,clipnorm=1.)


def gradient_penalty(discriminator,real_images,fake_images):
    alpha = tf.random.uniform([real_images.shape[0],1,1,1],0.,1.)
    interpolated = alpha * real_images + (1-alpha)*fake_images
    with tf.GradientTape() as tape :
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred,interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean((norm - 1.0)**2)
    return gradient_penalty


def discriminator_loss_unet(real_output, fake_output,discriminator,real_images,fake_images):
    # cross_entropy = tf.keras.losses.BinaryCrossentropy()
    # test 3 :
    #cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # --------------#
    # test avec Wasserstein loss
    lambda_gp = 10
    real_loss = wasserstein_loss(tf.ones_like(real_output), real_output)
    fake_loss = wasserstein_loss(-tf.ones_like(fake_output), fake_output)
    gp = gradient_penalty(discriminator,real_images,fake_images)
    total_loss = real_loss + fake_loss + lambda_gp * gp
    return total_loss

def clip_weights(model, clip_value=0.01):
    """Clip les poids du modèle dans un intervalle [-clip_value, clip_value]."""
    # Clip les variables d'entraînement du modèle
    for var in model.trainable_variables:
        clipped_var = tf.clip_by_value(var, -clip_value, clip_value)
        var.assign(clipped_var)  # Remplace la variable par la version clippée


@tf.function
def train_step_unet_gen(images_damaged,
                          generator,generator_optimizer,
                          discriminator):

    with tf.GradientTape() as gen_tape:
      generated_images = generator(images_damaged, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss_unet(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_generator = map(lambda grad : tf.clip_by_value(grad,-10.,10.),gradients_of_generator)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    clip_weights(generator)
    return float(gen_loss)

@tf.function
def train_step_unet_dis(images,images_damaged,
                          generator,
                          discriminator,discriminator_optimizer):

    with tf.GradientTape() as disc_tape:
      generated_images = generator(images_damaged, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      disc_loss = discriminator_loss_unet(real_output, fake_output, discriminator, images, generated_images)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_discriminator = map(lambda grad : tf.clip_by_value(grad,-10.,10.),gradients_of_discriminator)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    clip_weights(discriminator)
    return float(disc_loss)



def train_unet_model(data,data_damaged,
          generator,generator_optimizer,
          discriminator,discriminator_optimizer,
          epochs,batch_size = 128,
          chkpt = 1, workbook=True):

    progressive_output = []
    history_gen = []
    history_disc = []
    start = time.time()
    nb_batches = int(np.ceil(data.shape[0]/batch_size))
    data = data.astype('float32')

    nb_iter = 0
    for epoch in range(epochs):
        for i in range(nb_batches) :
            image_batch = data[batch_size*i : (i+1)*batch_size,:,:,:]
            image_damaged_batch = data_damaged[batch_size*i : (i+1)*batch_size,:,:,:]
            loss_gen = train_step_unet_gen(images_damaged=image_damaged_batch,
                                                        generator=generator,
                                                        generator_optimizer=generator_optimizer,
                                                        discriminator=discriminator)

            if nb_iter == 0 or nb_iter == 5 :
                loss_disc = train_step_unet_dis(images=image_batch,
                                                        images_damaged=image_damaged_batch,
                                                        generator=generator,
                                                        discriminator=discriminator,
                                                        discriminator_optimizer=discriminator_optimizer)

                nb_iter = 0

            nb_iter += 1
            history_disc.append(float(loss_disc))
            history_gen.append(float(loss_gen))

            print('epoch', epoch+1,'batch',i+1,'/', nb_batches, time.time()-start, 'loss_gen', float(loss_gen), 'loss_disc', float(loss_disc))

        if (epoch + 1)%chkpt == 0 or epoch == epochs - 1:
            # Show output pour faire une GIF:
            prediction = generator(np.expand_dims(data_damaged[0],axis=0), training=False)[0,:,:,:] # training = False ne modifie pas les poids des couches de neuronnes
            progressive_output.append(prediction)

            # save metrics
            metrics = pd.DataFrame({'history_generator_loss' : history_gen,'history_discriminator_loss' : history_disc})
            local_path_metrics = os.path.join(PATH_MODELS,'unet_model','metrics', f"{time.strftime('%Y%m%d-%H%M%S')}-epoch{epoch+1}.csv")
            path_dir_metrics = os.path.join(PATH_MODELS,'unet_model','metrics')
            if workbook :
                local_path_metrics = os.path.join('..',local_path_metrics)
                path_dir_metrics = os.path.join('..',path_dir_metrics)
            if not os.path.exists(path_dir_metrics) :
                os.makedirs(path_dir_metrics)
            metrics.to_csv(local_path_metrics)
            print('metrics saved')

            # save le model
            local_path_gen = os.path.join(PATH_MODELS,'unet_model','models','generator')
            local_path_dis = os.path.join(PATH_MODELS,'unet_model','models','discriminator')

            if workbook :
                local_path_gen = os.path.join('..',local_path_gen)
                local_path_dis = os.path.join('..',local_path_dis)

            path_gen = os.path.join(local_path_gen,f"{time.strftime('%Y%m%d-%H%M%S')}-gen-epoch{epoch+1}.h5")
            path_disc =os.path.join(local_path_dis,f"{time.strftime('%Y%m%d-%H%M%S')}-dis-epoch{epoch+1}.h5")

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
