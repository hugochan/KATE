'''
Created on Nov, 2016

@author: hugo

'''

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adadelta, Adam, Adagrad
from keras.models import load_model
from keras import regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras_utils import Dense_tied
import tensorflow as tf
# from keras.objectives import
# from keras.constraints import nonneg
# from utils import l1norm

sess = tf.InteractiveSession()
def weighted_binary_crossentropy(feature_weights):
    def loss(y_true, y_pred):
        try:
            x = K.binary_crossentropy(y_pred, y_true)
            y = tf.Variable(feature_weights.astype('float32'))
            # y2 = y_true / K.sum(y_true)
            # import pdb;pdb.set_trace()
            z = K.dot(x, y)
        except Exception as e:
            print e
            import pdb;pdb.set_trace()
        return z
        # return K.dot(K.binary_crossentropy(y_pred, y_true), feature_weights)
    return loss

class AutoEncoder(object):
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

    def fit(self, train_X, val_X, feature_weights=None, init_weights=None):
        n_feature = train_X[0].shape[1]
        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        if init_weights != None:
            encoded_layer = Dense(self.dim, init='normal', activation='relu', weights=[init_weights, np.random.randn(self.dim)/np.sqrt(n_feature)])
        else:
            encoded_layer = Dense(self.dim, init='normal', activation='relu')

        # add a Dense layer with a L1 activity regularizer
        # encoded_layer = Dense(self.dim, init='normal', activation='relu',
                        # activity_regularizer=regularizers.activity_l1(1e-2))
        encoded = encoded_layer(input_layer)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        # decoded = Dense(n_feature, init='normal', activation='sigmoid')(encoded)
        decoded = Dense_tied(n_feature, init='normal', activation='sigmoid', tied_to=encoded_layer)(encoded)

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

        optimizer = Adadelta(lr=2)
        # optimizer = Adam()
        # optimizer = Adagrad()
        self.autoencoder.compile(optimizer=optimizer, loss=weighted_binary_crossentropy(feature_weights)) # kld, binary_crossentropy, mse
        # self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
        checkpointer = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.autoencoder.fit(train_X[0], train_X[1],
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[checkpointer]
                        )

        return self

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
