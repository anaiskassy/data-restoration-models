# https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
import time
import os
from data_restoration.params import *
import matplotlib.pyplot as plt

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return tf.reduce_mean(y_true * y_pred)


# define the standalone critic model
def define_critic(in_shape=(16,16,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()
	# downsample to 8x8
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
	# downsample to 4x4
	model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
    # downsample to 2x2
	model.add(Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
    # downsample to 1x1
	model.add(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
	# scoring, linear activation
	model.add(Flatten())
	model.add(Dense(1))
	# compile model
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model


# define the standalone generator model
def define_generator(in_shape=(64,64,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	#---------------encoding---------------#
    # define model
	model = Sequential()
	# downsample to 32x32
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
	# downsample to 16x16
	model.add(Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
    # downsample to 8x8
	model.add(Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
    # downsample to 4x4
	model.add(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
    # downsample to 2x2
	model.add(Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(0.2))
    #---------------decoding---------------#
	# upsample to 4x4
	model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(ReLU(0.2))
	# upsample to 8x8
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(ReLU(0.2))
    # upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(ReLU(0.2))
    # upsample to 16x16
	model.add(Conv2DTranspose(64, (4,4), strides=(1,1), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(ReLU(0.2))
	# output 16x16x3
	model.add(Conv2D(3, (2,2), activation='tanh', padding='same', kernel_initializer=init))
	return model


# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model

def generate_indexes(dataset,n_samples) :
    # choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	return ix

# generate reals with class labels
def generate_real_outputs(expected_outputs,ix):
	X = expected_outputs[ix]
	# generate class labels, -1 for 'real'
	y = -np.ones((X.shape[0], 1))
	return X, y

# use the generator to generate fakes, with class labels
def generate_fake_images(generator, images_damaged_input,ix):
	# predict outputs
	X = generator.predict(images_damaged_input[ix])
	# create class labels with 1.0 for 'fake'
	y = np.ones((X.shape[0], 1))
	return X, y


# train the generator and critic
def train(g_model, c_model, gan_model, images_damaged_input, expected_outputs, n_epochs=10, batch_size=128, n_critic=5, chkpt = 10, workbook=False):
    start = time.time()
    # calculate the number of batches per training epoch
    nb_batches = int(np.ceil(images_damaged_input.shape[0]/batch_size))
    half_batch = int(nb_batches/2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # manually enumerate epochs
    for epoch in range(n_epochs):
        for i in range(nb_batches) :
            # update the critic more than the generator
            c1_tmp, c2_tmp = list(), list()
            expected_outputs_batch = expected_outputs[batch_size*i : (i+1)*batch_size,:,:,:]
            images_damaged_input_batch = images_damaged_input[batch_size*i : (i+1)*batch_size,:,:,:]
            for _ in range(n_critic):
                ix = generate_indexes(expected_outputs_batch,half_batch)
                X_real, y_real = generate_real_outputs(expected_outputs=expected_outputs_batch,ix=ix)
                # update critic model weights
                c_loss1 = c_model.train_on_batch(X_real, y_real)
                c1_tmp.append(c_loss1)
                # generate 'fake' examples
                X_fake, y_fake = generate_fake_images(g_model,images_damaged_input_batch,ix)
                # update critic model weights
                c_loss2 = c_model.train_on_batch(X_fake, y_fake)
                c2_tmp.append(c_loss2)
            # store critic loss
            c1_hist.append(np.mean(c1_tmp))
            c2_hist.append(np.mean(c2_tmp))

            # create inverted labels for the fake samples
            y_gan = -np.ones((images_damaged_input_batch.shape[0], 1))
            # update the generator via the critic's error
            g_loss = gan_model.train_on_batch(images_damaged_input_batch, y_gan)
            g_hist.append(g_loss)
            # summarize loss on this batch
            print('epoch', epoch+1,'batch',i+1,'/', nb_batches, time.time()-start, 'loss_gan', float(g_hist[-1]), 'loss_critic', float(c1_hist[-1]), float(c2_hist[-1]))

        if (epoch + 1)%chkpt == 0 or epoch == n_epochs - 1:
            # save metrics
            metrics = pd.DataFrame({'history_gan_loss' : g_hist,
                                    'history_critic_neg_loss' : c1_hist,
                                    'history_critic_pos_loss' : c2_hist})
            local_path_metrics = os.path.join(PATH_MODELS,'combined_model','metrics', f"{time.strftime('%Y%m%d-%H%M%S')}-epoch{epoch+1}.csv")
            path_dir_metrics = os.path.join(PATH_MODELS,'combined_model','metrics')
            if workbook :
                local_path_metrics = os.path.join('..',local_path_metrics)
                path_dir_metrics = os.path.join('..',path_dir_metrics)
            if not os.path.exists(path_dir_metrics) :
                os.makedirs(path_dir_metrics)
            metrics.to_csv(local_path_metrics)
            print('metrics saved')

            # save le model
            local_path_gen = os.path.join(PATH_MODELS,'combined_model','models','generator')
            local_path_dis = os.path.join(PATH_MODELS,'combined_model','models','discriminator')
            local_path_comb = os.path.join(PATH_MODELS,'combined_model','models','combined')

            if workbook :
                local_path_gen = os.path.join('..',local_path_gen)
                local_path_dis = os.path.join('..',local_path_dis)
                local_path_comb = os.path.join('..',local_path_comb)

            path_gen = os.path.join(local_path_gen,f"{time.strftime('%Y%m%d-%H%M%S')}-gen-epoch{epoch+1}.h5")
            path_disc =os.path.join(local_path_dis,f"{time.strftime('%Y%m%d-%H%M%S')}-dis-epoch{epoch+1}.h5")
            path_comb =os.path.join(local_path_comb,f"{time.strftime('%Y%m%d-%H%M%S')}-comb-epoch{epoch+1}.h5")

            if not os.path.exists(local_path_gen) :
                os.makedirs(local_path_gen)
            if not os.path.exists(local_path_dis) :
                os.makedirs(local_path_dis)
            if not os.path.exists(local_path_comb) :
                os.makedirs(local_path_comb)
            g_model.save(path_gen)
            c_model.save(path_disc)
            gan_model.save(path_comb)

            # client = storage.Client()
            # bucket = client.bucket(BUCKET_NAME_SMALL)
            # blob_gen = bucket.blob(f"model-unet/generator/{time.strftime('%Y%m%d-%H%M%S')}-gen-epoch{epoch+1}.h5")
            # blob_gen.upload_from_filename(path_gen)
            # blob_dis = bucket.blob(f"model-unet/discriminator/{time.strftime('%Y%m%d-%H%M%S')}-dis-epoch{epoch+1}.h5")
            # blob_dis.upload_from_filename(path_disc)

            print('models saved')


    # Generate after the final epoch
    predictions = g_model.predict(images_damaged_input[:100])
    plt.imshow((predictions[0]+1)/2);

    return g_hist, c1_hist , c2_hist, predictions










# # size of the latent space
# latent_dim = 50
# # create the critic
# critic = define_critic()
# # create the generator
# generator = define_generator(latent_dim)
# # create the gan
# gan_model = define_gan(generator, critic)
# # load image data
# dataset = load_real_samples()
# print(dataset.shape)
# # train model
# train(generator, critic, gan_model, dataset, latent_dim)
