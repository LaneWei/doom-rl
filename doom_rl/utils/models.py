from doom_rl.utils.core.model import DQNTfModel
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.layers import conv2d, dense
from tensorflow.nn import relu
from tensorflow.train import AdamOptimizer as Adam


class SimpleTfModel(DQNTfModel):
    def __init__(self, state_shape, nb_actions, learning_rate, preprocess_state_batch):
        super(SimpleTfModel, self).__init__(state_shape, nb_actions,
                                            learning_rate, preprocess_state_batch=preprocess_state_batch)

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

        self._max_q_values = tf.reduce_max(self.q_values, axis=1)
        self._best_action = tf.argmax(self.q_values, axis=1)
        self._action_q_values = tf.reduce_sum(self.q_values * tf.one_hot(self.a_input, self.nb_actions), axis=1)
        self._loss = tf.losses.mean_squared_error(self._action_q_values, self.target_q_values)
        opt = Adam(self.lr)
        self._train = opt.minimize(self._loss)
