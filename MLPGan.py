#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GAN implementation, idea extracted from [I. Goodfellow, 14]

__author__ = "José Ángel González"
__license__ = "GNU"
__version__ = "0.1"
__maintainer__ = "José Ángel González"
__email__ = "jogonba2@dsic.upv.es"
__status__ = "Testing"

from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model, Sequential
from keras.layers.core import Dropout, Activation, Flatten, Reshape
from keras.utils import np_utils, generic_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU as lrelu
import matplotlib.pyplot as plt
import theano
import os
import cv2
import numpy as np
np.random.seed(1337)  # for reproducibility

def discriminator_model(input_dim):
	discriminator_inputs = Input(shape=(input_dim,))
	discriminator = Dense(128)(discriminator_inputs)
	discriminator = lrelu()(discriminator)
	discriminator = Dropout(0.3)(discriminator)
	discriminator = Dense(128)(discriminator)
	discriminator = lrelu()(discriminator)
	discriminator = Dropout(0.3)(discriminator)
	discriminator_outputs = Dense(1, activation="sigmoid")(discriminator) 
	discriminator = Model(input=discriminator_inputs, output=discriminator_outputs)	
	return discriminator

def gan_model(latent_dim, discriminator):
	gan_inputs = Input(shape=(latent_dim,))
	gan = Dense(128)(gan_inputs)
	gan = lrelu()(gan)
	gan = Dense(128)(gan)
	gan = lrelu()(gan)
	gan_outputs = Dense(x_mnist.shape[1], activation="sigmoid")(gan)
	discriminator.trainable = False
	gan = discriminator(gan_outputs)
	gan = Model(input=gan_inputs, output=gan)
	gan.compile(loss='binary_crossentropy', optimizer="adam")
	return gan, gan_inputs, gan_outputs

# assert dim(generator(output))==dim(discriminator(input)) #
def generate_gan_model(input_dim, latent_dim):
	discriminator = discriminator_model(input_dim)
	discriminator.trainable = False
	gan, gan_inputs, gan_outputs = gan_model(latent_dim, discriminator)
	gan.compile(loss='binary_crossentropy', optimizer="adagrad")
	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy', optimizer="adagrad")
	generator = theano.function([gan_inputs], gan_outputs)
	return generator, discriminator, gan
	
def normalize_mnist(x_mnist): return x_mnist / 255.0

def generate_normal_noise(shape): return np.random.uniform(-1, 1, size=shape)

def prepare_real_data(n_samples):
	(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
	x_mnist = x_train_mnist[0:n_samples]
	x_mnist = normalize_mnist(x_mnist)
	x_mnist = np.reshape(x_mnist, (n_samples, x_mnist.shape[1]*x_mnist.shape[2]))
	del y_train_mnist, x_test_mnist, y_test_mnist
	return x_mnist

# There isn't hacks as: noise labels, round criteria, opposite labels, batch training etc. #
def train(generator, discriminator, gan, x, input_dim, latent_dim, epochs, batch_size, 
          view_evol=True, path="./", model_name="DCGAN.h5", n_channels=None, 
          width=None, height=None):
	if view_evol: fig = plt.figure()
	noise = generate_normal_noise((20, latent_dim))
	bar = generic_utils.Progbar(epochs)
	n_samples = x.shape[0]
	for i in range(epochs):
		loss_disc, loss_gan, loss = 0, 0, 0
		np.random.shuffle(x_mnist)
		for j in range(0, n_samples-batch_size, batch_size):
			if i%2==0:
				x_noise = generate_normal_noise((batch_size, latent_dim))
				y_noise = np.ones((batch_size, 1))
				gan_loss = gan.train_on_batch(x_noise, y_noise)
				loss_gan += gan_loss
			else:
				x_noise = generate_normal_noise((batch_size, latent_dim))
				x_generator = generator(x_noise)
				y_generator = np.zeros((batch_size, 1))
				x_real = x[j:j+batch_size].reshape((batch_size, input_dim))
				y_real = np.ones((batch_size, 1))
				x_train = np.concatenate((x_real, x_generator))
				y_train = np.concatenate((y_real, y_generator))
				disc_loss = discriminator.train_on_batch(x_train, y_train)
				loss_disc += disc_loss
		
		loss = loss_disc+loss_gan
		bar.update(i, values=[("Loss", loss), ("Gan loss", loss_gan), ("Disc loss", loss_disc)], force=True)
		if view_evol and i%2==0:
			plt.clf()
			x_pred   = generator(noise)
			for j in range(20):
				plt.subplot(4, 5, j+1)
				if n_channels==1 or n_channels==None:
					plt.imshow(x_pred[j].reshape((width, height)), cmap="gray")
				else:
					plt.imshow(x_pred[j].reshape((n_channels, width, height)))
				plt.axis('off')
			fig.canvas.draw()
			fig.savefig(path+"/Epoch"+str(i)+".png")		
	gan.save(model_name)


if __name__ == "__main__":
	x_mnist = prepare_real_data(500)
	n_channels = 1
	width  = 28
	height = 28
	input_dim = x_mnist.shape[1]
	latent_dim = 100
	batch_size = 128
	epochs = 100
	generator, discriminator, gan = generate_gan_model(input_dim, latent_dim)
	train(generator, discriminator, gan, x_mnist, input_dim, latent_dim, epochs, 
	      batch_size, True, "./AAA", "DCGAN.h5", n_channels, width, height)
