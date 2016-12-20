'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.models import load_model
from keras import regularizers
from keras import objectives
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
import keras.backend as K
import tensorflow as tf

from .keras_utils import Dense_tied
# from keras.constraints import nonneg
# from utils import l1norm


class VarAutoEncoder(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        batch_size :

        """

    def __init__(self, dim, nb_epoch=50, batch_size=100, model_save_path='./'):
        self.latent_dim = dim
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.model_save_path = model_save_path

    def fit(self, train_X, val_X, feature_weights=None):

        self.n_feature = train_X[0].shape[1]
        intermediate_dim = 1024
        self.epsilon_std = 1.0

        input_layer = Input(batch_shape=(self.batch_size, self.n_feature))
        hidden_layer1 = Dense(intermediate_dim, batch_input_shape=(None, self.n_feature), init='glorot_normal', activation='sigmoid')
        h1 = hidden_layer1(input_layer)
        self.z_mean = Dense(self.latent_dim, batch_input_shape=(None, intermediate_dim), init='glorot_normal')(h1)
        self.z_log_var = Dense(self.latent_dim, batch_input_shape=(None, intermediate_dim), init='glorot_normal')(h1)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        latent_layer = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, batch_input_shape=(None, self.latent_dim), init='glorot_normal', activation=PReLU())
        h_decoded = decoder_h(latent_layer)
        decoder_mean = Dense_tied(self.n_feature, batch_input_shape=(None, intermediate_dim), init='glorot_normal', \
                    activation='sigmoid', tied_to=hidden_layer1)
        x_decoded_mean = decoder_mean(h_decoded)

        self.vae = Model(input_layer, x_decoded_mean)

        # optimizer = Adadelta(lr=1.)
        optimizer = 'rmsprop'

        self.vae.compile(optimizer=optimizer, loss=self.weighted_vae_loss(feature_weights))
        # self.vae.compile(optimizer=optimizer, loss=self.vae_loss)

        self.vae.fit(train_X[0], train_X[1],
                shuffle=True,
                nb_epoch=self.nb_epoch,
                batch_size=self.batch_size,
                validation_data=(val_X[0], val_X[1]),
                callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                            # ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, verbose=0),
                        ]
                )


        # build a model to project inputs on the latent space
        self.encoder = Model(input_layer, self.z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.decoder = Model(decoder_input, _x_decoded_mean)

        return self

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = self.n_feature * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def weighted_vae_loss(self, feature_weights):
        def loss(y_true, y_pred):
            try:
                x = K.binary_crossentropy(y_pred, y_true)
                y = tf.Variable(feature_weights.astype('float32'))
                # y2 = y_true / K.sum(y_true)
                # import pdb;pdb.set_trace()
                xent_loss = K.dot(x, y)
                kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            except Exception as e:
                print e
                import pdb;pdb.set_trace()
            return xent_loss + kl_loss
        return loss

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
