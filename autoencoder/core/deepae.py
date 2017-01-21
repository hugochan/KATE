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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.core import Activation
# from keras.layers.normalization import BatchNormalization

from ..utils.keras_utils import Dense_tied, weighted_binary_crossentropy, KCompetitive
from ..utils.io_utils import dump_json, load_json

class DeepAutoEncoder(object):
    """Deep AutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        batch_size :

        """

    def __init__(self, input_size, dim, comp_topk=None, \
        weights_file=None, model_save_path='./'):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.model_save_path = model_save_path

        self.build(weights_file)

    def build(self, weights_file=None):
        h1_dim = 256

        # this is our input placeholder
        input_layer = Input(shape=(self.input_size,))

        # "encoded" is the encoded representation of the input
        h1_layer = Dense(h1_dim, init='glorot_normal', activation='tanh')
        encoded_layer = Dense(self.dim, init='glorot_normal', activation='tanh')

        encoded = h1_layer(input_layer)

        if self.comp_topk:
            encoded = KCompetitive(self.comp_topk)(encoded)
            print 'add k-competitive layer'

        encoded = encoded_layer(encoded)
        if self.comp_topk:
            encoded = KCompetitive(self.comp_topk)(encoded)
            print 'add k-competitive layer'

        # "decoded" is the lossy reconstruction of the input
        decoder_layer = Dense_tied(h1_dim, init='glorot_normal', activation='tanh', tied_to=encoded_layer)
        rev_h1_layer = Dense_tied(self.input_size, init='glorot_normal', activation='sigmoid', tied_to=h1_layer)
        decoded = decoder_layer(encoded)

        if self.comp_topk:
            decoded = KCompetitive(self.comp_topk)(decoded)
            print 'add k-competitive layer'

        decoded = rev_h1_layer(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_layer, output=decoded)

        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_layer, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.dim,))

        # create the decoder model
        self.decoder = Model(input=encoded_input, output=rev_h1_layer(decoder_layer(encoded_input)))

        if not weights_file is None:
            self.autoencoder.load_weights(weights_file, by_name=True)
            print 'Loaded pretrained weights'

    def fit(self, train_X, val_X, nb_epoch=50, batch_size=100, feature_weights=None):
        print 'Training autoencoder'
        optimizer = Adadelta(lr=1.5)
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
                                    # ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, verbose=0),
                        ]
                        )

        return self
