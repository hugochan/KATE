'''
Created on Nov, 2016

@author: hugo

'''

import numpy as np
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adadelta, Adam, Adagrad
from keras.models import load_model
from keras import regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras_utils import Dense_tied, weighted_binary_crossentropy, KSparseScheduler, KSparse
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
# from keras.objectives import
# from keras.constraints import nonneg
# from utils import l1norm

# sess = tf.InteractiveSession()


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
        n_feature = train_X[0].shape[1]
        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        if init_weights is None:
            encoded_layer = Dense(self.dim, init='glorot_normal', activation='sigmoid', name='Encoded_Layer')
            # encoded_layer = Dense(self.dim, init='glorot_normal')
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
            encoded = KSparse(sparse_topk, sparse_alpha if sparse_alpha else 1)(encoded)
            print 'add k-sparse layer'
        # encoded = Dropout(.2)(encoded)




        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        # decoded = Dense(n_feature, init='glorot_normal', activation='sigmoid')(encoded)
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
            print 'using feature weights'
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
                                    # KSparseScheduler(sparsity_level, start_k, end_k, step_k, alpha, sparsity_level),
                        ]
                        )

        return self

    def fit_deep(self, train_X, val_X, feature_weights=None):
        n_feature = train_X[0].shape[1]
        h1_dim = 1024

        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # # "encoded" is the encoded representation of the input
        # h1_layer = Dense(h1_dim, init='glorot_normal', activation='sigmoid')
        # encoded_layer = Dense(self.dim, init='glorot_normal', activation='relu')

        # encoded = h1_layer(input_layer)
        # encoded = encoded_layer(encoded)

        # # "decoded" is the lossy reconstruction of the input
        # decoder_layer = Dense_tied(h1_dim, init='glorot_normal', activation='relu', tied_to=encoded_layer)
        # rev_h1_layer = Dense_tied(n_feature, init='glorot_normal', activation='sigmoid', tied_to=h1_layer)
        # decoded = decoder_layer(encoded)
        # decoded = rev_h1_layer(decoded)




        # "encoded" is the encoded representation of the input
        h1_layer = Dense(h1_dim, init='glorot_normal')
        encoded = h1_layer(input_layer)
        encoded = BatchNormalization((h1_dim,))(encoded)
        encoded = Activation('sigmoid')(encoded)


        encoded_layer = Dense(self.dim, init='glorot_normal')
        encoded = encoded_layer(encoded)
        encoded = BatchNormalization((self.dim,))(encoded)
        encoded = Activation('relu')(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoder_layer = Dense_tied(h1_dim, init='glorot_normal', tied_to=encoded_layer)
        decoded = decoder_layer(encoded)
        decoded = BatchNormalization((h1_dim,))(decoded)
        decoded = Activation('relu')(decoded)



        rev_h1_layer = Dense_tied(n_feature, init='glorot_normal', tied_to=h1_layer)
        decoded = rev_h1_layer(decoded)
        # decoded = BatchNormalization((n_feature,))(decoded)
        decoded = Activation('sigmoid')(decoded)

        # # "decoded" is the lossy reconstruction of the input
        # decoder_layer = Dense_tied(h1_dim, init='glorot_normal', activation='relu', tied_to=encoded_layer)
        # rev_h1_layer = Dense_tied(n_feature, init='glorot_normal', activation='sigmoid', tied_to=h1_layer)
        # decoded = decoder_layer(encoded)
        # decoded = rev_h1_layer(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_layer, output=decoded)


        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_layer, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.dim,))

        # create the decoder model
        self.decoder = Model(input=encoded_input, output=rev_h1_layer(decoder_layer(encoded_input)))

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
                                    # ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, verbose=0),
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

    def kSparse(self, X, **kwargs):
        k = int(X.get_shape()[1]) - kwargs['sparsity']['topk']
        values, indices = tf.nn.top_k(-X, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)])  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(X), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
        res = X + to_reset

        return res

    def save_all(self, model_dict):
        for k, v in model_dict:
            k.save(v)

    def save_mod(self, mod_file):
        self.autoencoder.save(mod_file)

        return self

    def load_mod(self, mod_file):
        return load_model(mod_file)


def demo():
    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))[:10000]
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))[:1000]

    ae = AutoEncoder(dim=32, nb_epoch=50, batch_size=100).fit([x_train, x_train], [x_test, x_test])
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_x = ae.encoder.predict(x_test)
    decoded_x = ae.decoder.predict(encoded_x)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    demo()
