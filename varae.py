'''
Created on Nov, 2016

@author: hugo

'''

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.models import load_model
from keras import regularizers
# from keras.constraints import nonneg
# from utils import l1norm
import keras.backend as K
from keras import objectives



class VarAutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        batch_size :

        """

    def __init__(self, dim, nb_epoch=50, batch_size=100):
        self.dim = dim
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

    def fit(self, train_X, val_X):

        self.n_feature = train_X[0].shape[1]
        self.latent_dim = self.dim
        intermediate_dim = 256
        self.epsilon_std = 1.0

        x = Input(batch_shape=(self.batch_size, self.n_feature))
        h = Dense(intermediate_dim, activation='sigmoid')(x)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(self.n_feature, activation='softmax')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        self.vae = Model(x, x_decoded_mean)
        self.vae.compile(optimizer='adadelta', loss=self.vae_loss)

        self.vae.fit(train_X[0], train_X[1],
                shuffle=True,
                nb_epoch=self.nb_epoch,
                batch_size=self.batch_size,
                validation_data=(val_X[0], val_X[1]))


        # build a model to project inputs on the latent space
        self.encoder = Model(x, self.z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.decoder = Model(decoder_input, _x_decoded_mean)

        return self

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.n_feature * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                  std=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def save_all(self, model_dict):
        for k, v in model_dict:
            k.save(v)

    def save_mod(self, mod_file):
        self.autoencoder.save(mod_file)

        return self

    def load_mod(self, mod_file):
        return load_model(mod_file)
