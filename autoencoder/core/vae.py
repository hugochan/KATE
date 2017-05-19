'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.models import load_model as load_keras_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.advanced_activations import PReLU
import keras.backend as K

from ..utils.keras_utils import Dense_tied, KCompetitive, KCompetitive, CustomModelCheckpoint
from ..utils.io_utils import dump_json, load_json


class VarAutoEncoder(object):
    """VarAutoEncoder for topic modeling.

        Parameters
        ----------
        dim : dimensionality of encoding space.

        nb_epoch :

        """

    def __init__(self, input_size, dim, comp_topk=None, ctype=None, epsilon_std=1.0, save_model='best_model'):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.epsilon_std = epsilon_std
        self.save_model = save_model

        self.build()

    def build(self):
        act = 'tanh'
        input_layer = Input(shape=(self.input_size,))
        hidden_layer1 = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        h1 = hidden_layer1(input_layer)

        # if self.comp_topk and self.comp_topk[0] != -1:
        #     print 'add k-competitive layer'
        #     h1 = KCompetitive(self.comp_topk[0], self.ctype)(h1)

        self.z_mean = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)
        self.z_log_var = Dense(self.dim[1], kernel_initializer='glorot_normal')(h1)

        if self.comp_topk and self.comp_topk[1] != -1:
            print 'add k-competitive layer'
            self.z_mean = KCompetitive(self.comp_topk[1], self.ctype)(self.z_mean)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        encoded = Lambda(self.sampling, output_shape=(self.dim[1],))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.dim[0], kernel_initializer='glorot_normal', activation=act)
        h_decoded = decoder_h(encoded)
        decoder_mean = Dense_tied(self.input_size, activation='sigmoid', tied_to=hidden_layer1)
        x_decoded_mean = decoder_mean(h_decoded)

        self.vae = Model(outputs=x_decoded_mean, inputs=input_layer)
        # build a model to project inputs on the latent space
        self.encoder = Model(outputs=self.z_mean, inputs=input_layer)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.dim[1],))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.decoder = Model(outputs=_x_decoded_mean, inputs=decoder_input)

    def fit(self, train_X, val_X, nb_epoch=50, batch_size=100):
        print 'Training variational autoencoder'
        optimizer = Adadelta(lr=2.)
        self.vae.compile(optimizer=optimizer, loss=self.vae_loss)

        self.vae.fit(train_X[0], train_X[1],
                shuffle=True,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_data=(val_X[0], val_X[1]),
                callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                            EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                            CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        ]
                )

        return self

    def vae_loss(self, x, x_decoded_mean):
        xent_loss =  K.sum(K.binary_crossentropy(x_decoded_mean, x), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)

        return xent_loss + kl_loss

    # def weighted_vae_loss(self, feature_weights):
    #     def loss(y_true, y_pred):
    #         try:
    #             x = K.binary_crossentropy(y_pred, y_true)
    #             y = tf.Variable(feature_weights.astype('float32'))
    #             # y2 = y_true / K.sum(y_true)
    #             # import pdb;pdb.set_trace()
    #             xent_loss = K.dot(x, y)
    #             kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
    #         except Exception as e:
    #             print e
    #             import pdb;pdb.set_trace()
    #         return xent_loss + kl_loss
    #     return loss

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.dim[1]), mean=0.,\
                                  stddev=self.epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon

def save_vae_model(model, model_file):
    model.encoder.save(model_file)

def load_vae_model(model_file):
    return load_keras_model(model_file, custom_objects={"KCompetitive": KCompetitive})
