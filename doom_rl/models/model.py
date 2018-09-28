import tensorflow as tf


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

    Args:
        learning_rate: The learning rate. (used by an optimizer)
        preprocess_state_batch: A function that takes a batch of states as input,
        then perform pre-processing on the batch of states and return the processed batch.
    """
    def __init__(self, learning_rate, preprocess_state_batch=lambda x: x):
        self.preprocess_state_batch = preprocess_state_batch
        self.lr = learning_rate

    def save_weights(self, save_path):
        """
        Save the weights of this model to a file.

        Args:
            save_path: The file path to save the weights of this model.
        """
        pass

    def load_weights(self, load_path):
        """
        Load the weights of this model from a file.

        Args:
            load_path: The file path from which the weights of this model is to be loaded.
        """
        pass

    def train(self, state, action, target_q):
        """
        Perform a learning step to train this model.

        Args:
            state: A batch of states.
            action: A batch of actions taken.
            target_q: Target q values.

        Returns:
            The calculated loss of this training step.
        """
        pass

    def get_best_action(self, state):
        """
        Given the current state as input, get the best action (with the highest q value)
        according to this model's network.

        Args:
            state: The current state.

        Returns:
            The the action with the highest q value.
        """
        pass

    def get_q_values(self, state):
        """
        Get the q values of all actions at a given state.

        Args:
            state: The current state.

        Returns:
            A list containing the q values of all actions.
        """
        raise NotImplementedError('Function get_q_values not implemented.')

    def get_max_q_values(self, state):
        """
        Get the highest q value at a given state.

        Args:
            state: The given state.

        Returns:
            The highest q value at the given state.
        """
        raise NotImplementedError('Function get_max_q_values not implemented.')


class DQNTfModel(Model):
    """
    This is a half-implemented DQN model based on pure tensorflow. All abstract methods defined
    in base class `Model` are implemented.
    You can easily extend this class by just implementing `_build_network()` method in your subclass.

    Args:
        state_shape: The shape of input states. (to be modified)
        nb_actions: The number of actions that the agent can perform. (to be modified)
        learning_rate: The learning rate.
    """

    def __init__(self, state_shape, nb_actions, learning_rate, **kwargs):
        super(DQNTfModel, self).__init__(learning_rate, **kwargs)
        self.state_shape = state_shape
        self.nb_actions = nb_actions

        # Tensorflow session
        self._session = None

        # State input of the network
        self.s_input = tf.placeholder(tf.float32, shape=[None] + list(self.state_shape), name='State')

        # Action input, indicating the actions chosen under self.s_input
        self.a_input = tf.placeholder(tf.int32, shape=[None], name='Action')

        # Target q values, for calculating the loss
        self.target_q_values = tf.placeholder(tf.float32, shape=[None], name='TargetQ')

        # Output q values, must be used in _build_network() method as the output of the network
        self.q_values = None

        # Optimizer,
        self._optimizer = None

        # Build the network
        self._build_network()

        # Check whether the definition for self.q_values and self._optimizer are provided
        assert self.q_values is not None, 'The output of the network is not defined.'
        assert self._optimizer is not None, 'No optimizer is provided.'

        # Operation nodes defined by _build_network() method
        self._max_q_values = tf.reduce_max(self.q_values, axis=1)
        self._best_action = tf.argmax(self.q_values, axis=1)
        self._action_q_values = tf.reduce_sum(self.q_values * tf.one_hot(self.a_input, self.nb_actions), axis=1)
        self._loss = tf.losses.mean_squared_error(self._action_q_values, self.target_q_values)
        self._train = self._optimizer.minimize(self._loss)

    def save_weights(self, save_path):
        tf.train.Saver().save(self.session, save_path)

    def load_weights(self, load_path):
        tf.train.Saver().restore(self.session, load_path)

    def train(self, state, action, target_q):
        state = self.preprocess_state_batch(state)
        l, _ = self.session.run([self._loss, self._train],
                                {self.s_input: state, self.target_q_values: target_q, self.a_input: action})
        return l

    def get_best_action(self, state):
        state = self.preprocess_state_batch(state)
        return self.session.run(self._best_action, {self.s_input: state})[0]

    def get_q_values(self, state):
        state = self.preprocess_state_batch(state)
        return self.session.run(self.q_values, {self.s_input: state})

    def get_max_q_values(self, state):
        state = self.preprocess_state_batch(state)
        return self.session.run(self._max_q_values, {self.s_input: state})

    def _build_network(self):
        """
        Build the model's network. (Example implementation in doom_rl.utils.models.SimpleTfModel)

        To implement this method you should:
        1. Build the structure of your network. Use self.s_input as your network's input and self.q_values
           as your network's output.
        2. The loss of the network is already defined as self._loss and the training operation defined as
           self._train = self._optimizer.minimize(self._loss), however, you need to provide the definition
           for self._optimizer in this method. (Adam optimizer, RMSProp optimizer etc.)
        3. Use self.lr as the optimizer's learning_rate
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
