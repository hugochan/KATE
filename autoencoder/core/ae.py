'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
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
import tensorflow as tf

from ..utils.keras_utils import Dense_tied, weighted_binary_crossentropy, KSparse


class AutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        batch_size :

        """

    def __init__(self, dim, nb_epoch=50, batch_size=100, model_save_path='./'):
        self.dim = dim
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.model_save_path = model_save_path

    def fit(self, train_X, val_X, sparse_topk=None, sparse_alpha=None, feature_weights=None, init_weights=None, weights_file=None):
        print 'running autoencoder'
        n_feature = train_X[0].shape[1]
        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        if init_weights is None:
            encoded_layer = Dense(self.dim, init='glorot_normal', activation='sigmoid', name='Encoded_Layer')
            # encoded_layer = Dense(self.dim, init='glorot_normal', name='Encoded_Layer')
        else:
            encoded_layer = Dense(self.dim, activation='sigmoid', weights=init_weights, name='Encoded_Layer')

        # add a Dense layer with a L1 activity regularizer
        # encoded_layer = Dense(self.dim, init='normal', activation='relu',
                        # activity_regularizer=regularizers.activity_l1(1e-2))
        # input_layer = Dropout(.5)(input_layer)
        encoded = encoded_layer(input_layer)


        # start_k = 200
        # end_k = 40
        # step_k = 2
        # alpha = 1.0
        # sparsity_level = {'topk': tf.Variable(70, name='topk')}
        # sparsity_level = {'topk': self.dim}
        # import pdb;pdb.set_trace()
        # encoded = Lambda(self.kSparse, output_shape=(self.dim,), arguments={'sparsity': sparsity_level})(encoded)
        if sparse_topk:
            print 'add k-sparse layer'
            encoded = KSparse(sparse_topk, sparse_alpha if sparse_alpha else 1)(encoded)
        # encoded = Dropout(.2)(encoded)
        # encoded = Activation('sigmoid')(encoded)



        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        # decoded = Dense(n_feature, init='glorot_normal', activation='sigmoid', name='Decoded_Layer')(encoded)
        decoded = Dense_tied(n_feature, init='glorot_normal', activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

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
        if feature_weights is None:
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
        else:
            print 'using weighted loss'
            self.autoencoder.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(feature_weights)) # kld, binary_crossentropy, mse

        if not weights_file is None:
            self.autoencoder.load_weights(weights_file, by_name=True)

        self.autoencoder.fit(train_X[0], train_X[1],
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, verbose=0),
                        ]
                        )

        return self

    def fit_deepfit(self, train_X, val_X, sparse_topk=None, sparse_alpha=None, feature_weights=None, init_weights=None, weights_file=None):
        print 'running deep autoencoder'
        n_feature = train_X[0].shape[1]
        h1_dim = 512

        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        h1_layer = Dense(h1_dim, init='glorot_normal', activation='sigmoid')
        encoded_layer = Dense(self.dim, init='glorot_normal', activation='sigmoid')

        encoded = h1_layer(input_layer)

        if sparse_topk:
            encoded = KSparse(sparse_topk, sparse_alpha if sparse_alpha else 1)(encoded)
            print 'add k-sparse layer'

        encoded = encoded_layer(encoded)

        if sparse_topk:
            encoded = KSparse(sparse_topk, sparse_alpha if sparse_alpha else 1)(encoded)
            print 'add k-sparse layer'

        # "decoded" is the lossy reconstruction of the input
        decoder_layer = Dense_tied(h1_dim, init='glorot_normal', activation='sigmoid', tied_to=encoded_layer)
        rev_h1_layer = Dense_tied(n_feature, init='glorot_normal', activation='sigmoid', tied_to=h1_layer)
        decoded = decoder_layer(encoded)

        if sparse_topk:
            decoded = KSparse(sparse_topk, sparse_alpha if sparse_alpha else 1)(decoded)
            print 'add k-sparse layer'

        decoded = rev_h1_layer(decoded)
        if sparse_topk:
            decoded = KSparse(sparse_topk, sparse_alpha if sparse_alpha else 1)(decoded)
            print 'add k-sparse layer'



        # Batch Normalization

        # # "encoded" is the encoded representation of the input
        # h1_layer = Dense(h1_dim, init='glorot_normal')
        # encoded = h1_layer(input_layer)
        # encoded = BatchNormalization((h1_dim,))(encoded)
        # encoded = Activation('sigmoid')(encoded)


        # encoded_layer = Dense(self.dim, init='glorot_normal')
        # encoded = encoded_layer(encoded)
        # encoded = BatchNormalization((self.dim,))(encoded)
        # encoded = Activation('relu')(encoded)

        # # "decoded" is the lossy reconstruction of the input
        # decoder_layer = Dense_tied(h1_dim, init='glorot_normal', tied_to=encoded_layer)
        # decoded = decoder_layer(encoded)
        # decoded = BatchNormalization((h1_dim,))(decoded)
        # decoded = Activation('relu')(decoded)


        # rev_h1_layer = Dense_tied(n_feature, init='glorot_normal', tied_to=h1_layer)
        # decoded = rev_h1_layer(decoded)
        # decoded = BatchNormalization((n_feature,))(decoded)
        # decoded = Activation('sigmoid')(decoded)


        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_layer, output=decoded)


        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_layer, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.dim,))

        # create the decoder model
        self.decoder = Model(input=encoded_input, output=rev_h1_layer(decoder_layer(encoded_input)))

        optimizer = Adadelta(lr=1.0)
        # optimizer = Adam()
        # optimizer = Adagrad()
        if feature_weights is None:
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
        else:
            print 'using weighted loss'
            self.autoencoder.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(feature_weights)) # kld, binary_crossentropy, mse

        # self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
        self.autoencoder.fit(train_X[0], train_X[1],
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, verbose=0),
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
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, verbose=0),
                        ]
                        )

        return self
