'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
from os import path
import numpy as np
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adadelta, Adam, Adagrad
from keras.models import load_model
from keras import regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.core import Activation
# from keras.layers.normalization import BatchNormalization
import tensorflow as tf

from ..utils.keras_utils import Dense_tied, weighted_binary_crossentropy, KCompetitive
from ..utils.io_utils import dump_json, load_json

class AutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        batch_size :

        """

    def __init__(self, input_size, dim, comp_topk=None, weights_file=None):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk

        self.build(weights_file)

    def build(self, weights_file=None):
        # this is our input placeholder
        input_layer = Input(shape=(self.input_size,))

        # "encoded" is the encoded representation of the input
        encoded_layer = Dense(self.dim, init='glorot_normal', activation='tanh', name='Encoded_Layer')
        encoded = encoded_layer(input_layer)

        if self.comp_topk:
            print 'add k-competitive layer'
            encoded = KCompetitive(self.comp_topk)(encoded)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        # decoded = Dense(self.input_size, init='glorot_normal', activation='sigmoid', name='Decoded_Layer')(encoded)
        decoded = Dense_tied(self.input_size, init='glorot_normal', activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_layer, output=decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_layer, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        if not weights_file is None:
            self.autoencoder.load_weights(weights_file, by_name=True)
            print 'Loaded pretrained weights'

    def fit(self, train_X, val_X, nb_epoch=50, batch_size=100, feature_weights=None):
        print 'Training autoencoder'
        optimizer = Adadelta(lr=2.5)
        # optimizer = Adam()
        # optimizer = Adagrad()
        if feature_weights is None:
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
        else:
            print 'Using weighted loss'
            self.autoencoder.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(feature_weights)) # kld, binary_crossentropy, mse

        self.autoencoder.fit(train_X[0], train_X[1],
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                        ]
                        )

        return self

    def fit_batchnorm(self, train_X, val_X, feature_weights=None, init_weights=None):
        n_feature = train_X[0].shape[1]
        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        if not init_weights is None:
            encoded_layer = Dense(self.dim, init='glorot_normal', weights=init_weights)
        else:
            encoded_layer = Dense(self.dim, init='glorot_normal')

        encoded = encoded_layer(input_layer)
        encoded = BatchNormalization((self.dim,))(encoded)
        encoded = Activation('sigmoid')(encoded)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        # decoded = Dense(n_feature, init='glorot_normal', activation='sigmoid')(encoded)
        decoded = Dense_tied(n_feature, init='glorot_normal', activation='sigmoid', tied_to=encoded_layer)(encoded)
        # decoded = Dense_tied(n_feature, init='glorot_normal', tied_to=encoded_layer)(encoded)
        # decoded = BatchNormalization((self.dim,))(decoded)
        # decoded = Activation('sigmoid')(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_layer, output=decoded)


        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_layer, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        optimizer = Adadelta(lr=1.5)
        # optimizer = Adam()
        # optimizer = Adagrad()
        self.autoencoder.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(feature_weights)) # kld, binary_crossentropy, mse
        # self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
        self.autoencoder.fit(train_X[0], train_X[1],
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto')
                        ]
                        )

        return self

def save_model(model, arch_file, weights_file):
    arch = {'input_size': model.input_size,
            'dim': model.dim,
            'comp_topk': model.comp_topk}
    model.autoencoder.save_weights(weights_file)
    dump_json(arch, arch_file)

def load_model(model, arch_file, weights_file):
    arch = load_json(arch_file)
    ae = model(arch['input_size'], arch['dim'], comp_topk=arch['comp_topk'], weights_file=weights_file)

    return ae
