'''
Created on Nov, 2016

@author: hugo

'''

from keras.layers import Input, Dense, Layer, Activation
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.constraints import nonneg, Constraint
from utils import l1norm
import keras.backend as K


def linear(x):
    return x

class Antirectifier(Layer):
    '''This is the combination of a sample-wise
    L2 normalization with the concatenation of the
    positive part of the input with the negative part
    of the input. The result is a tensor of samples that are
    twice as large as the input samples.
    It can be used in place of a ReLU.
    # Input shape
        2D tensor of shape (samples, n)
    # Output shape
        2D tensor of shape (samples, 2*n)
    # Theoretical justification
        When applying ReLU, assuming that the distribution
        of the previous output is approximately centered around 0.,
        you are discarding half of your input. This is inefficient.
        Antirectifier allows to return all-positive outputs like ReLU,
        without discarding any data.
        Tests on MNIST show that Antirectifier allows to train networks
        with twice less parameters yet with comparable
        classification accuracy as an equivalent ReLU-based network.
    '''
    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, x, mask=None):
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)


class NonNegUnitL1Norm(Constraint):
    '''Constrain the weights incident to each hidden unit to have unit norm.
    # Arguments
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).
    '''
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p / (K.epsilon() + K.sum(p, axis=self.axis, keepdims=True))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}


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
        n_feature = train_X[0].shape[1]
        # this is our input placeholder
        input_layer = Input(shape=(n_feature,))

        # "encoded" is the encoded representation of the input
        encoded = Dense(self.dim, init='normal', activation='relu', bias=True)(input_layer)

        # add a Dense layer with a L1 activity regularizer
        # from keras import regularizers
        # encoded = Dense(self.dim, activation='relu',
                        # activity_regularizer=regularizers.activity_l1(1e-5))(input_layer)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        decoded = Dense(n_feature, init='normal', activation='softmax',
                        # W_constraint=NonNegUnitL1Norm(axis=1), # nonneg(),
                        W_constraint=nonneg(),
                        # b_constraint=nonneg(),
                        bias=False)(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input=input_layer, output=decoded)

        # self.autoencoder.add(Activation('softmax'))

        # this model maps an input to its encoded representation
        self.encoder = Model(input=input_layer, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        optimizer = Adadelta(lr=3.)
        # optimizer = Adam()
        self.autoencoder.compile(optimizer=optimizer, loss='kld') # kld, binary_crossentropy, mse

        self.autoencoder.fit(train_X[0], train_X[1],
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]))

        return self

    def save_mod(self, mod_file):
        self.autoencoder.save(mod_file)

        return self

    def load_mod(self, mod_file):
        self.autoencoder = load_model(mod_file)

        return self

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
