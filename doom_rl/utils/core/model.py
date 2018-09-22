import tensorflow as tf
import tensorflow.keras as tfk


class Model:
    """
    Base class for implementing a q_value based model that encapsulates a network
    (tensorflow graphs, keras model, etc.) and operations on this model.

    You need to implement the following methods:
        * `save_weights`
        * `load_weights`
        * `train`
        * `get_best_action`
        * `get_q_values` (optional)
        * `get_max_q_values` (optional)

    # Argument:
        * `preprocess_state_batch` A function that takes a batch of states as input,
        then perform pre-processing on the batch of states and return the processed batch.
    """
    def __init__(self, preprocess_state_batch=lambda x: x):
        self.preprocess_state_batch = preprocess_state_batch

    def save_weights(self, save_path):
        """
        Save the weights of this model to a file.

        :param save_path: The file path to save the weights of this model.
        """
        pass

    def load_weights(self, load_path):
        """
        Load the weights of this model from a file.

        :param load_path: The file path from which the weights of this model is to be loaded.
        """
        pass

    def train(self, state, action, target_q):
        """
        Perform a training step on this model.

        :param state: A batch of states.
        :param action: A batch of actions taken.
        :param target_q: Target q values.
        :return: The loss of this training step.
        """
        pass

    def get_best_action(self, state):
        """
        Given the current state as input, get the best action (with the highest q value)
        according to this model's network.

        :param state: The current state.
        :return: The the action with the highest q value.
        """
        pass

    def get_q_values(self, state):
        """
        Get all q values at a given state.
        :param state: The current state.
        :return: A list containing all q values.
        """
        raise NotImplementedError('Function get_q_values not implemented.')

    def get_max_q_values(self, state):
        """
        Get the highest q value at a given state.

        :param state: The current state.
        :return: The highest q value.
        """
        raise NotImplementedError('Function get_max_q_values not implemented.')


class DQNTfModel(Model):
    """
    This is a half-implemented DQN model based on pure tensorflow. All abstract methods defined
    in base class `Model` are implemented.
    You can easily extend this class by providing a `_build_network()` method in your subclass.

    # Arguments:
        * `state_shape` The shape of input states.
        * `nb_actions` The number of actions that the agent can perform.
        * `preprocess_state_batch` A function that takes a batch of states as input,
        then perform pre-processing on the batch of states and return the processed batch.
    """

    def __init__(self, state_shape, nb_actions, learning_rate, **kwargs):
        super(DQNTfModel, self).__init__(**kwargs)
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.lr = learning_rate

        # State input of the network
        self.s_input = tf.placeholder(tf.float32, shape=[None] + list(self.state_shape), name='State')

        # Action input, indicating the actions chosen under self.s_input
        self.a_input = tf.placeholder(tf.int32, shape=[None], name='Action')

        # Target q values, for calculating the loss
        self.target_q_values = tf.placeholder(tf.float32, shape=[None], name='TargetQ')

        # Output q values, must be used in _build_network() method as the output of the network
        self.q_values = None

        # Operation nodes defined by _build_network() method
        self._max_q_values = None
        self._best_action = None
        self._action_q_values = None
        self._loss = None
        self._session = None

        # Training operation
        self._train = None

        self._build_network()

    def save_weights(self, save_path):
        tf.train.Saver().save(self.session, save_path)

    def load_weights(self, load_path):
        tf.train.Saver().restore(self.session, load_path)

    def train(self, state, action, target_q):
        assert self._train is not None, 'Training operation can not be None.'

        state = self.preprocess_state_batch(state)
        l, _ = self.session.run([self._loss, self._train],
                                {self.s_input: state, self.target_q_values: target_q, self.a_input: action})
        return l

    def get_best_action(self, state):
        assert self.q_values is not None, 'The output of the network can not be None.'

        state = self.preprocess_state_batch(state)
        return self.session.run(self._best_action, {self.s_input: state})[0]

    def get_q_values(self, state):
        assert self.q_values is not None, 'The output of the network can not be None.'

        state = self.preprocess_state_batch(state)
        return self.session.run(self.q_values, {self.s_input: state})

    def get_max_q_values(self, state):
        assert self.q_values is not None, 'The output of the network can not be None.'

        state = self.preprocess_state_batch(state)
        return self.session.run(self._max_q_values, {self.s_input: state})

    def _build_network(self):
        """
        Build the model's network.
        To implement this method you should:
        1. Build the structure of your network. Use self.s_input as your network's input and self.q_values
           as your network's output.
        2. The loss of the network is already defined as self._loss, however, you need to provide an
           optimizer and the definition of self._train for training operation.
        3. self._max_q_values = tf.reduce_max(self.q_values, axis=1)
           self._best_action = tf.argmax(self.q_values, axis=1)
           self._action_q_values = tf.reduce_sum(self.q_values * tf.one_hot(self.a_input, self.nb_actions), axis=1)
           self._loss = tf.losses.mean_squared_error(self._action_q_values, self.target_q_values)
        """
        pass

    @property
    def session(self):
        if self._session is None:
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        return self._session


class DQNKerasModel(Model):
    pass
