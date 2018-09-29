from doom_rl.models.model import DQNTfModel
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.layers import conv2d, dense
from tensorflow.nn import relu


class SimpleTfModel(DQNTfModel):
    def __init__(self, state_shape, nb_actions, preprocess_state_batch):
        super(SimpleTfModel, self).__init__(state_shape, nb_actions,
                                            preprocess_state_batch=preprocess_state_batch)

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

        self.q_values = dense(fc1, self.nb_actions, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.constant_initializer(0.01))
