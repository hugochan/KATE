'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
from keras.layers import Dense
from keras.callbacks import Callback
import keras.backend as K
from keras.engine import Layer
import tensorflow as tf


class KSparseScheduler(Callback):
    '''Callback that schedules the sparsity level over epochs as follows:
    Suppose we are aiming for a sparsity level of k = 15.
    Then, we start off with a large sparsity level (e.g. k = 100)
    for which the k-sparse autoencoder can train all the hidden units.
    We then linearly decrease the sparsity level from k = 100 to k = 15
    over the first half of the epochs.
    This initializes the autoencoder in a good regime,
    for which all of the hidden units have a significant chance of being picked.
    Then, we keep k = 15 for the second half of the epochs.
    With this scheduling, we can train all of the filters, even for low sparsity levels.

    ref: Makhzani, Alireza, and Brendan Frey. "k-Sparse Autoencoders." arXiv preprint arXiv:1312.5663 (2013).

    Here, we adopt a slightly different strategy. We start from a starting sparsity level,
    and then decrease in every epoch until we hit the ending sparsity level.
    Note that when testing, we use a different sparsity level (alpha * end_k).

    Parameters
    ----------
    topk_dict : sparsity level
    start_k : starting sparsity level
    end_k : ending sparsity level
    step_k : step size of decreasing sparsity level
    alpha : use alpha * end_k at testing time
    '''

    def __init__(self, topk_dict, start_k, end_k, step_k, alpha, test):
        self.topk_dict = topk_dict
        self.key = topk_dict.keys()[0]
        self.start_k = start_k
        self.end_k = end_k
        self.step_k = step_k
        self.alpha = alpha
        self.test = test
        super(KSparseScheduler, self).__init__()

    def on_train_begin(self, logs={}):
        # import pdb;pdb.set_trace()
        # self.topk_dict[self.key] = int(self.start_k)
        self.topk_dict[self.key].assign(int(self.start_k))
        # self.topk_dict[self.key] = tf.mul(tf.constant(1, tf.int32), self.start_k)
        # print 'on_train_begin'
        # print self.topk_dict[self.key]

    def on_train_end(self, logs={}):
        # self.topk_dict[self.key] = int(self.alpha * self.end_k)
        self.topk_dict[self.key] = tf.cast(self.topk_dict[self.key], tf.float32)
        self.topk_dict[self.key] = tf.mul(self.topk_dict[self.key], self.alpha)
        self.topk_dict[self.key] = tf.cast(self.topk_dict[self.key], tf.int32)
        # print 'on_train_end'
        # print self.topk_dict

    def on_epoch_end(self, epoch, logs={}):
        # self.topk_dict[self.key] = max(self.topk_dict[self.key] - self.step_k, self.end_k)
        # prev = self.topk_dict[self.key]
        self.topk_dict[self.key] = tf.maximum(self.topk_dict[self.key] - self.step_k, self.end_k)
        # with tf.Session() as sess:
        #     init = tf.global_variables_initializer()
        #     sess.run(init)
        #     import pdb;pdb.set_trace()
        #     print sess.run(self.topk_dict[self.key])
        # print 'on_epoch_end'
        # print self.topk_dict

class KSparse(Layer):
    '''Applies K-Sparse layer.

    # Arguments

    # References
        - Makhzani, Alireza, and Brendan Frey. "k-Sparse Autoencoders." arXiv preprint arXiv:1312.5663 (2013).
    '''
    def __init__(self, topk, alpha, **kwargs):
        self.topk = topk
        self.alpha = alpha
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KSparse, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # res = K.in_train_phase(self.kSparse(x, self.topk), self.kSparse(x, int(self.alpha * self.topk)))
        res = K.in_train_phase(self.kSparse(x, self.topk), x)
        return res

    def get_config(self):
        config = {'topk': self.topk, 'alpha': self.alpha}
        base_config = super(KSparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def kSparse(self, x, topk):
        print 'run k-sparse with compensation'
        dim = int(x.get_shape()[1])
        if topk > dim:
            print 'Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim)
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)])  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)



        tmp = tf.reduce_sum(to_reset) / topk
        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tmp), [-1]), default_value=0., validate_indices=False)

        res = tf.add(x, to_reset)

        # preserve lost engery
        # method 1) scale up
        # res = float(dim) / topk * res

        # method 2) add complement
        res = tf.sub(res, tmp)

        return res

    def regular_kSparse(self, x, topk):
        print 'run regular k-sparse'
        dim = int(x.get_shape()[1])
        if topk > dim:
            print 'Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim)
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)])  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        res = tf.add(x, to_reset)

        return res

class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(output_dim=output_dim, init=init,
                 activation=activation, weights=weights,
                 W_regularizer=W_regularizer, b_regularizer=b_regularizer,
                 activity_regularizer=activity_regularizer,
                 W_constraint=W_constraint, b_constraint=b_constraint,
                 bias=bias, input_dim=input_dim, **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.W in self.trainable_weights:
            self.trainable_weights.remove(self.W)


    def call(self, x, mask=None):
        # Use tied weights
        self.W = K.transpose(self.tied_to.W)
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

def weighted_binary_crossentropy(feature_weights):
    def loss(y_true, y_pred):
        # try:
        #     x = K.binary_crossentropy(y_pred, y_true)
        #     # y = tf.Variable(feature_weights.astype('float32'))
        #     # z = K.dot(x, y)
        #     y_true = tf.pow(y_true + 1e-5, .75)
        #     y2 = tf.div(y_true, tf.reshape(K.sum(y_true, 1), [-1, 1]))
        #     z = K.sum(tf.mul(x, y2), 1)
        # except Exception as e:
        #     print e
        #     import pdb;pdb.set_trace()
        # return z
        return K.dot(K.binary_crossentropy(y_pred, y_true), tf.Variable(feature_weights.astype('float32')))
    return loss
