'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import os
import numpy as np
from keras.layers import Dense
from keras.callbacks import Callback
import keras.backend as K
from keras.engine import Layer
import tensorflow as tf
from keras import initializers
import warnings

from ..testing.visualize import heatmap
from .op_utils import unitmatrix

def contractive_loss(model, lam=1e-4):
    def loss(y_true, y_pred):
        ent_loss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

        W = K.variable(value=model.encoder.get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.encoder.output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return ent_loss + contractive
    return loss


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
        return K.dot(K.binary_crossentropy(y_pred, y_true), K.variable(feature_weights.astype('float32')))
    return loss


class KCompetitive(Layer):
    '''Applies K-Competitive layer.

    # Arguments
    '''
    def __init__(self, topk, ctype, **kwargs):
        self.topk = topk
        self.ctype = ctype
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KCompetitive, self).__init__(**kwargs)

    def call(self, x):
        if self.ctype == 'ksparse':
            return K.in_train_phase(self.kSparse(x, self.topk), x)
        elif self.ctype == 'kcomp':
            return K.in_train_phase(self.k_comp_tanh(x, self.topk), x)
        else:
            warnings.warn("Unknown ctype, using no competition.")
            return x

    def get_config(self):
        config = {'topk': self.topk, 'ctype': self.ctype}
        base_config = super(KCompetitive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def k_comp_sigm(self, x, topk):
    #     print 'run k_comp_sigm'
    #     dim = int(x.get_shape()[1])
    #     if topk > dim:
    #         warnings.warn('topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
    #         topk = dim

    #     values, indices = tf.nn.top_k(x, topk) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

    #     # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    #     my_range = tf.expand_dims(tf.range(0, K.shape(indices)[0]), 1)  # will be [[0], [1]]
    #     my_range_repeated = tf.tile(my_range, [1, topk])  # will be [[0, 0], [1, 1]]

    #     full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    #     full_indices = tf.reshape(full_indices, [-1, 2])

    #     to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

    #     batch_size = tf.to_float(tf.shape(x)[0])
    #     tmp = 1 * batch_size * tf.reduce_sum(x - to_reset, 1, keep_dims=True) / topk

    #     res = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tmp), [-1]), default_value=0., validate_indices=False)

    #     return res

    def k_comp_tanh(self, x, topk, factor=6.26):
        print 'run k_comp_tanh'
        dim = int(x.get_shape()[1])
        # batch_size = tf.to_float(tf.shape(x)[0])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2

        values, indices = tf.nn.top_k(P, topk / 2) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, topk / 2])  # will be [[0, 0], [1, 1]]
        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)


        values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0., validate_indices=False)


        # 1)
        # res = P_reset - N_reset
        # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk

        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)

        # 2)
        # factor = 0.
        # factor = 2. / topk
        P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True) # 6.26
        N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]), default_value=0., validate_indices=False)
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]), default_value=0., validate_indices=False)

        res = P_reset - N_reset

        return res

    # def k_comp_tanh_strict(self, x, topk):
    #     print 'run k_comp_tanh_strict'
    #     dim = int(x.get_shape()[1])
    #     # batch_size = tf.to_float(tf.shape(x)[0])
    #     if topk > dim:
    #         warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
    #         topk = dim

    #     x_abs = tf.abs(x)
    #     P = (x + x_abs) / 2 # positive part of x
    #     N = (x - x_abs) / 2 # negative part of x

    #     values, indices = tf.nn.top_k(x_abs, topk) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
    #     # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    #     my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
    #     my_range_repeated = tf.tile(my_range, [1, topk])  # will be [[0, 0], [1, 1]]
    #     full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    #     full_indices = tf.reshape(full_indices, [-1, 2])
    #     x_topk_mask = tf.sparse_to_dense(full_indices, tf.shape(x), tf.ones([tf.shape(full_indices)[0], ], tf.float32), default_value=0., validate_indices=False)

    #     P_select = tf.multiply(P, x_topk_mask)
    #     N_select = tf.multiply(-N, x_topk_mask)

    #     zero = tf.constant(0., dtype=tf.float32)
    #     P_indices = tf.cast(tf.where(tf.not_equal(P_select, zero)), tf.int32)
    #     N_indices = tf.cast(tf.where(tf.not_equal(N_select, zero)), tf.int32)
    #     P_mask = tf.sparse_to_dense(P_indices, tf.shape(x), tf.ones([tf.shape(P_indices)[0], ], tf.float32), default_value=0., validate_indices=False)
    #     N_mask = tf.sparse_to_dense(N_indices, tf.shape(x), tf.ones([tf.shape(N_indices)[0], ], tf.float32), default_value=0., validate_indices=False)

    #     alpha = 10.
    #     P_complement = alpha * tf.reduce_sum(P - P_select, 1, keep_dims=True)# / tf.cast(tf.shape(P_indices)[0], tf.float32) # 6.26
    #     N_complement = alpha * tf.reduce_sum(-N - N_select, 1, keep_dims=True)# / tf.cast(tf.shape(N_indices)[0], tf.float32)

    #     P_reset = tf.multiply(tf.add(P_select, P_complement), P_mask)
    #     N_reset = tf.multiply(tf.add(N_select, N_complement), N_mask)
    #     res = P_reset - N_reset

    #     return res

    def kSparse(self, x, topk):
        print 'run regular k-sparse'
        dim = int(x.get_shape()[1])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.stack([my_range_repeated, indices], axis=2) # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        res = tf.add(x, to_reset)

        return res


class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                 activation=activation, use_bias=use_bias,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)


    def call(self, x, mask=None):
        # Use tied weights
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)

class CustomModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, custom_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.custom_model = custom_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('CustomModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        model = self.custom_model
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            model.save_weights(filepath, overwrite=True)
                        else:
                            model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    model.save_weights(filepath, overwrite=True)
                else:
                    model.save(filepath, overwrite=True)

class VisualWeights(Callback):
    def __init__(self, save_path, per_epoch=15):
        super(VisualWeights, self).__init__()
        self.per_epoch = per_epoch
        self.filename, self.ext = os.path.splitext(save_path)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        if epoch % self.per_epoch == 0:
            weights = self.model.get_weights()[0]
            # weights /= np.max(np.abs(weights))
            weights = unitmatrix(weights, axis=0) # normalize
            # weights[np.abs(weights) < 1e-2] = 0
            heatmap(weights.T, '%s_%s%s'%(self.filename, epoch, self.ext))
