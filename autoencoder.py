'''
Created on Nov, 2016

@author: hugo

'''

from keras.layers import Input, Dense
from keras.models import Model


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

    def fit(self, train_X, val_X):
        n_feature = train_X.shape[1]
        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        encoded = Dense(self.dim, activation='relu')(input_layer)

        # from keras import regularizers
        # # add a Dense layer with a L1 activity regularizer
        # encoded = Dense(self.dim, activation='relu',
        #                 activity_regularizer=regularizers.activity_l1(10e-5))(input_layer)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(n_feature, activation='sigmoid')(encoded)

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

        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.autoencoder.fit(train_X, train_X,
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(val_X, val_X))

        return self

    def save_mod(self, mod_file):
        self.autoencoder.save(mod_file)

        return self

    def load_mod(self, mod_file):
        self.autoencoder = load_model(mod_file)

        return self


if __name__ == '__main__':
    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))[:10000]
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))[:1000]

    ae = AutoEncoder(dim=32, nb_epoch=50, batch_size=100).fit(x_train, x_test)
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
