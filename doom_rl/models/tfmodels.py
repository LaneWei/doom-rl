from doom_rl.models.model import DqnTfModel
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.layers import conv2d, dense
from tensorflow.nn import relu


class SimpleTfModel(DqnTfModel):
    def __init__(self, state_shape, nb_actions, discount_factor, **kwargs):
        super(SimpleTfModel, self).__init__(state_shape, nb_actions, discount_factor, **kwargs)

    def _build_network(self):
        conv1 = conv2d(self.s_input, 24, 6, strides=(3, 3), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer1')
        conv2 = conv2d(conv1, 32, 3, strides=(2, 2), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer2')
        conv_flat = flatten(conv2)
        fc1 = dense(conv_flat, 128, activation=relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.constant_initializer(0.01), name='FullyConnected1')

        output_layer = dense(fc1, self.nb_actions, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.01), name='QValues')
        return output_layer


# A toy model which requires less computation
class SimplerTfModel(DqnTfModel):
    def __init__(self, state_shape, nb_actions, discount_factor):
        super(SimplerTfModel, self).__init__(state_shape, nb_actions, discount_factor)

    def _build_network(self):
        conv1 = conv2d(self.s_input, 6, 6, strides=(3, 3), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer1')
        conv2 = conv2d(conv1, 12, 3, strides=(2, 2), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer2')
        conv_flat = flatten(conv2)
        output_layer = dense(conv_flat, self.nb_actions, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.01), name='QValues')
        return output_layer
