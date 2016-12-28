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
from keras import initializations

class MyModelCheckpoint(Callback):
    '''Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.
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
    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('MyModelCheckpoint mode %s is unknown, '
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
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
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
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

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

class KCompetitive(Layer):
    '''Applies K-Competitive layer.

    # Arguments
    '''
    def __init__(self, topk, **kwargs):
        self.topk = topk
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KCompetitive, self).__init__(**kwargs)

    def add_weight(self, shape, initializer, name=None,
                   trainable=True,
                   regularizer=None,
                   constraint=None):
        '''Adds a weight variable to the layer.
        # Arguments:
            shape: The shape tuple of the weight.
            initializer: An Initializer instance (callable).
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            regularizer: An optional Regularizer instance.
        '''
        initializer = initializations.get(initializer)
        weight = initializer(shape, name=name)
        if regularizer is not None:
            self.add_loss(regularizer(weight))
        if constraint is not None:
            self.constraints[weight] = constraint
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight

    # def build(self, input_shape):
    #     # assert len(input_shape) >= 2
    #     # input_dim = input_shape[-1]
    #     # self.input_dim = input_dim
    #     # import pdb;pdb.set_trace()
    #     self.alpha = self.add_weight(( ), initializer='one')
    #     self.built = True

    def call(self, x, mask=None):
        # res = K.in_train_phase(self.k_comp(x, self.topk), self.k_comp(x, self.topk)*1.5)
        res = K.in_train_phase(self.k_comp_abs(x, self.topk), x)
        return res

    def get_config(self):
        config = {'topk': self.topk}
        base_config = super(KCompetitive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def k_comp(self, x, topk):
        print 'run k_comp'
        dim = int(x.get_shape()[1])
        if topk > dim:
            print 'Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim)
            topk = dim

        values, indices = tf.nn.top_k(x, topk) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, topk])  # will be [[0, 0], [1, 1]]

        full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)])  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)

        batch_size = tf.to_float(tf.shape(x)[0])
        tmp = 1 * batch_size * tf.reduce_sum(x - to_reset, 1, keep_dims=True) / topk

        res = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tmp), [-1]), default_value=0., validate_indices=False)

        return res

    def k_comp_abs(self, x, topk):
        print 'run k_comp_abs'
        dim = int(x.get_shape()[1])
        batch_size = tf.to_float(tf.shape(x)[0])
        if topk > dim:
            print 'Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim)
            topk = dim

        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2

        values, indices = tf.nn.top_k(P, topk / 2) # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, topk / 2])  # will be [[0, 0], [1, 1]]
        full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)])  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0., validate_indices=False)


        values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
        full_indices2 = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices2, 2)])  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0., validate_indices=False)


        # 1)
        # res = P_reset - N_reset
        # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk

        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)

        # 2)
        P_tmp = 1 * batch_size * tf.reduce_sum(P - P_reset, 1, keep_dims=True) / (topk / 2)
        N_tmp = 1 * batch_size * tf.reduce_sum(-N - N_reset, 1, keep_dims=True) / (topk - topk / 2)
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]), default_value=0., validate_indices=False)
        N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]), default_value=0., validate_indices=False)


        res = P_reset - N_reset

        return res

    def kSparse(self, x, topk):
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
