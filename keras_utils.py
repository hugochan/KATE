from keras.layers import Dense
import keras.backend as K

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
