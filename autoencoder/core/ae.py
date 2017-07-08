'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adadelta
from keras.models import load_model as load_keras_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ..utils.keras_utils import Dense_tied, KCompetitive, contractive_loss, CustomModelCheckpoint


class AutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        """

    def __init__(self, input_size, dim, comp_topk=None, ctype=None, save_model='best_model'):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.save_model = save_model

        self.build()

    def build(self):
        # this is our input placeholder
        input_layer = Input(shape=(self.input_size,))

        # "encoded" is the encoded representation of the input
        if self.ctype == None:
            act = 'sigmoid'
        elif self.ctype == 'kcomp':
            act = 'tanh'
        elif self.ctype == 'ksparse':
            act = 'linear'
        else:
            raise Exception('unknown ctype: %s' % self.ctype)
        encoded_layer = Dense(self.dim, activation=act, kernel_initializer="glorot_normal", name="Encoded_Layer")
        encoded = encoded_layer(input_layer)

        if self.comp_topk:
            print 'add k-competitive layer'
            encoded = KCompetitive(self.comp_topk, self.ctype)(encoded)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        decoded = Dense_tied(self.input_size, activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(outputs=decoded, inputs=input_layer)

        # this model maps an input to its encoded representation
        self.encoder = Model(outputs=encoded, inputs=input_layer)

        # create a placeholder for an encoded input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(outputs=decoder_layer(encoded_input), inputs=encoded_input)

    def fit(self, train_X, val_X, nb_epoch=50, batch_size=100, contractive=None):
        optimizer = Adadelta(lr=2.)
        # optimizer = Adam()
        # optimizer = Adagrad()
        if contractive:
            print 'Using contractive loss, lambda: %s' % contractive
            self.autoencoder.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print 'Using binary crossentropy'
            self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse

        self.autoencoder.fit(train_X[0], train_X[1],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        ]
                        )

        return self

def save_ae_model(model, model_file):
    model.save(model_file)

def load_ae_model(model_file):
    return load_keras_model(model_file, custom_objects={"KCompetitive": KCompetitive})
