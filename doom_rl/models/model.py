import tensorflow as tf


class Model:
    """
    Base class for implementing a q_value based model that encapsulates a network
    (tensorflow graphs, keras model, etc.) and operations on this model.

    Note: From a model's perspective, actions are represented by integers and are different
    from the real actions that one agent can perform.

    You need to implement the following methods:
        * `compile`
        * `save_weights`
        * `load_weights`
        * `train`
        * `get_best_action`
        * `get_q_values` (optional)
        * `get_max_q_values` (optional)

    Args:
        process_state_batch: A function that takes a batch of states as input,
        then pre-processes the batch of states and returns the processed batch.
    """
    def __init__(self, process_state_batch=lambda x: x):
        self.process_state_batch = process_state_batch

    def compile(self, learning_rate, optimizer='Adam', **kwargs):
        """
        Compile this model and provide an optimizer.
        A model has to be compiled before performing any operations.

        Args:
            learning_rate: The learning rate of the optimizer.
            optimizer: A string that specifies the type of the optimizer. Three optimizers
            are available, including: 'Adam' (default), 'RMSProp', 'SGD'.
        """
        raise NotImplementedError()

    def save_weights(self, save_path):
        """
        Save the weights of this model to file.

        Args:
            save_path: The file path to save the weights of this model.
        """
        raise NotImplementedError()

    def load_weights(self, load_path):
        """
        Load the weights of this model from file.

        Args:
            load_path: The file path from which the weights of this model is to be loaded.
        """
        raise NotImplementedError()

    def train(self, state, action, target_q):
        """
        Perform a learning step to train this model.

        Args:
            state: A batch of states.
            action: A batch of actions.
            target_q: Target q values.

        Returns:
            The calculated loss of this training step.
        """
        raise NotImplementedError()

    def get_best_action(self, state):
        """
        Given the current state as input, get the best action (with the highest q value)
        according to this model's network.

        Args:
            state: The current state.

        Returns:
            An integer representing the the action with the highest q value.
        """
        raise NotImplementedError()

    def get_q_values(self, state):
        """
        Get the q values of all actions at a given state.

        Args:
            state: The current state.

        Returns:
            A list containing the q values of all actions.
        """
        pass

    def get_max_q_values(self, state):
        """
        Get the highest q value at a given state.

        Args:
            state: The given state.

        Returns:
            The highest q value at the given state.
        """
        pass


class DQNTfModel(Model):
    """
    This is a half-implemented DQN model based on pure tensorflow. All abstract methods defined
    in base class `Model` are implemented.
    You can easily extend this class by just implementing `_build_network()` method in your subclass.

    Args:
        state_shape: The shape of input states, which indicates the input shape of this model's
        input layer.
        nb_actions: The number of actions that the agent can perform, which indicates the number of
        units in this model's output layer.
    """

    def __init__(self, state_shape, nb_actions, **kwargs):
        super(DQNTfModel, self).__init__(**kwargs)
        self.state_shape = state_shape
        self.nb_actions = nb_actions

        # Tensorflow session
        self._session = None

        # State input of the network
        self.s_input = None

        # Action input, indicating the actions chosen under self.s_input
        self.a_input = None

        # Target q values, for calculating the loss
        self.target_q_values = None

        # Output q values, must be used in _build_network() method as the output of the network
        self.q_values = None

        # Optimizer
        self._optimizer = None

        # Operation nodes defined by _build_network() method
        self._max_q_values = None
        self._best_action = None
        self._action_q_values = None
        self._loss = None
        self._train = None

        # summary
        # ...

        self._model_created = False

    def save_weights(self, save_path):
        tf.train.Saver().save(self.session, save_path)

    def load_weights(self, load_path):
        tf.train.Saver().restore(self.session, load_path)

    def train(self, state, action, target_q):
        state = self.process_state_batch(state)
        l, _ = self.session.run([self._loss, self._train],
                                {self.s_input: state, self.target_q_values: target_q, self.a_input: action})
        return l

    def get_best_action(self, state):
        state = self.process_state_batch(state)
        return self.session.run(self._best_action, {self.s_input: state})[0]

    def get_q_values(self, state):
        state = self.process_state_batch(state)
        return self.session.run(self.q_values, {self.s_input: state})

    def get_max_q_values(self, state):
        state = self.process_state_batch(state)
        return self.session.run(self._max_q_values, {self.s_input: state})

    def compile(self, learning_rate, optimizer='Adam', **kwargs):
        optimizer = optimizer.lower()
        optimizers = {'adam': tf.train.AdamOptimizer,
                      'rmsprop': tf.train.RMSPropOptimizer,
                      'sgd': tf.train.GradientDescentOptimizer}
        if optimizer not in optimizers:
            raise KeyError('Invalid optimizer {}.'.format(optimizer))

        if not self._model_created:
            self._create_placeholders()
            self._build_network()
            # Check whether the definition for self.q_values is provided by self._build_network
            assert self.q_values is not None, 'The output of the network is not defined.'

            self._create_operations()
            self._model_created = True

        self._optimizer = optimizers[optimizer](learning_rate=learning_rate, **kwargs)
        self._train = self._optimizer.minimize(self._loss)

        # Start session
        print(self.session.run(tf.constant("Model compiled.")))

    def _create_placeholders(self):
        assert self.s_input is None
        assert self.a_input is None
        assert self.target_q_values is None
        self.s_input = tf.placeholder(tf.float32, shape=[None] + list(self.state_shape), name='State')
        self.a_input = tf.placeholder(tf.int32, shape=[None], name='Action')
        self.target_q_values = tf.placeholder(tf.float32, shape=[None], name='TargetQ')

    def _create_operations(self):
        assert self._max_q_values is None
        assert self._best_action is None
        assert self._action_q_values is None
        assert self._loss is None
        self._max_q_values = tf.reduce_max(self.q_values, axis=1)
        self._best_action = tf.argmax(self.q_values, axis=1)
        self._action_q_values = tf.reduce_sum(self.q_values * tf.one_hot(self.a_input, self.nb_actions), axis=1)
        self._loss = tf.losses.mean_squared_error(self._action_q_values, self.target_q_values)

    def _build_network(self):
        """
        Build the model's network. (Example implementation in doom_rl.models.tfmodels.SimpleTfModel)

        To implement this method you should:
            Build the structure of your network. Use self.s_input as your network's input and self.q_values
            as your network's output. The number of units in the output layer is self.nb_actions.
        """
        pass

    @property
    def session(self):
        if not self._model_created:
            raise RuntimeError('A model must be compiled before performing any operations.')
        if self._session is None:
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        return self._session


class DQNKerasModel(Model):
    pass
