import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


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
    """

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

    def train(self, states, actions, rewards, next_states, terminates):
        """
        Perform a learning step to train this model.

        Args:
            states: A batch of states.
            actions: A batch of actions.
            rewards: A batch of rewards.
            next_states: A batch of next states.
            terminates: Termination flags.

        Returns:
            The calculated loss of this training step.
        """
        raise NotImplementedError()

    def get_best_action(self, states):
        """
        Given the current state as input, get the best action (with the highest q value)
        according to this model's network.

        Args:
            states: A batch of current states.

        Returns:
            A list of actions, represented by integers, with the highest q value.
        """
        raise NotImplementedError()

    def get_q_values(self, states):
        """
        Get the q values of all actions at a given state.

        Args:
            states: A batch of current states.

        Returns:
            A list containing the q values of all actions.
        """
        pass

    def get_max_q_values(self, states):
        """
        Get the highest q value at a given state.

        Args:
            states: A batch of states.

        Returns:
            The corresponding highest q values at the given states.
        """
        pass


class DqnTfModel(Model):
    """
    This is a half-implemented DQN model based on pure tensorflow. All abstract methods defined
    in base class `Model` are implemented.
    You can easily extend this class by just implementing `_build_network()` method in your subclass.

    Args:
        state_shape: The shape of input states, which indicates the input shape of this model's
        input layer.
        nb_actions: The number of actions that the agent can perform, which indicates the number of
        units in this model's output layer.
        enable_ddqn: Enable double dqn.
    """

    def __init__(self, state_shape, nb_actions, discount_factor, update_steps=1000, enable_ddqn=True, **kwargs):
        super(DqnTfModel, self).__init__(**kwargs)
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.gamma = discount_factor
        self.update_steps = update_steps
        self.ddqn = enable_ddqn
        self.steps = 0

        # Tensorflow session
        self._session = None

        # State input of the network, a placeholder in the computation graph
        self.s_input = tf.placeholder(tf.float32, shape=[None] + list(self.state_shape), name='In_State')

        # Action input, a placeholder, indicating the actions chosen under self.s_input
        self.a_input = tf.placeholder(tf.int32, shape=[None], name='In_Action')

        # Target q values for updating the training network
        self.train_target_q = tf.placeholder(tf.float32, shape=[None], name='Train_TargetQ')

        # An operation for updating the target network
        self.update_target_network = None

        self._train_network = {
            'scope_name': 'TrainNetwork',

            # Output q values of the training network
            'q_values': None,
            'optimizer': None,
            'loss': None,
            'Op': {
                'max_q_values': None,
                'best_actions': None,
                'action_q_values': None,
                'train': None
            }
        }

        self._target_network = {
            'scope_name': 'TargetNetwork',

            # Output q values of the target network
            'q_values': None,
            'Op': {
                'max_q_values': None,
                'action_q_values': None,
            }
        }

        # summary
        # ...

        self._model_created = False

    def save_weights(self, save_path):
        tf.train.Saver().save(self.session, save_path)

    def load_weights(self, load_path):
        tf.train.Saver().restore(self.session, load_path)

    def update_target(self):
        """
        Force to update the target network.
        """
        self.session.run(self.update_target_network)

    def train(self, states, actions, rewards, next_states, terminates):
        self.steps = (self.steps + 1) % self.update_steps
        if self.ddqn:
            action_selections = self.session.run(self._train_network['Op']['best_actions'],
                                                 feed_dict={self.s_input: next_states})
            eval_q = self.session.run(self._target_network['Op']['action_q_values'],
                                      feed_dict={self.s_input: next_states, self.a_input: action_selections})
            train_target = rewards + self.gamma * eval_q * (1 - terminates)
        else:
            eval_q = self.session.run(self._target_network['Op']['max_q_values'],
                                      feed_dict={self.s_input: next_states})
            train_target = rewards + self.gamma * eval_q * (1 - terminates)
        l, _ = self.session.run([self._train_network['loss'], self._train_network['Op']['train']],
                                {self.s_input: states, self.train_target_q: train_target, self.a_input: actions})

        if self.steps == 0:
            self.update_target()
        return l

    def get_best_action(self, states):
        return self.session.run(self._train_network['Op']['best_action'], {self.s_input: states})

    def get_q_values(self, states):
        return self.session.run(self._train_network['q_values'], {self.s_input: states})

    def get_max_q_values(self, states):
        return self.session.run(self._train_network['Op']['max_q_values'], {self.s_input: states})

    def compile(self, learning_rate, optimizer='Adam', **kwargs):
        optimizer = optimizer.lower()
        optimizers = {'adam': tf.train.AdamOptimizer,
                      'rmsprop': tf.train.RMSPropOptimizer,
                      'sgd': tf.train.GradientDescentOptimizer}
        if optimizer not in optimizers:
            raise KeyError('Invalid optimizer {}.'.format(optimizer))

        if not self._model_created:
            # Train network
            with tf.variable_scope(self._train_network['scope_name']):
                with tf.variable_scope('NetLayers'):
                    self._train_network['q_values'] = self._build_network()

            # Target network
            with tf.variable_scope(self._target_network['scope_name']):
                with tf.variable_scope('NetLayers'):
                    self._target_network['q_values'] = self._build_network()

            self._create_operations()
            self._model_created = True

        opt = optimizers[optimizer](learning_rate, name=optimizer, **kwargs)
        with tf.variable_scope(self._train_network['scope_name']):
            self._train_network['optimizer'] = opt
            self._train_network['Op']['train'] = opt.minimize(self._train_network['loss'], name="TrainOp")

        # This also starts the session
        tf.summary.FileWriter('log', self.session.graph)

    def _create_operations(self):
        train_op = self._train_network['Op']
        train_q = self._train_network['q_values']
        with tf.variable_scope(self._train_network['scope_name']):
            train_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope=self._train_network['scope_name'])
            train_op['max_q_values'] = tf.reduce_max(train_q, axis=1, name='MaxQValues')
            train_op['best_actions'] = tf.argmax(train_q, axis=1, name='BestActions')
            train_op['action_q_values'] = tf.reduce_sum(train_q * tf.one_hot(self.a_input, self.nb_actions),
                                                        axis=1, name='ActionQValues')
            self._train_network['loss'] = tf.losses.mean_squared_error(train_op['action_q_values'], self.train_target_q)

        target_op = self._target_network['Op']
        target_q = self._target_network['q_values']
        with tf.name_scope(self._target_network['scope_name']):
            target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=self._target_network['scope_name'])
            target_op['max_q_values'] = tf.reduce_max(target_q, axis=1, name='MaxQValues')
            target_op['action_q_values'] = tf.reduce_sum(target_q * tf.one_hot(self.a_input, self.nb_actions),
                                                         axis=1, name='ActionQValues')

        self.update_target_network = [tf.assign(w_train, w_target, validate_shape=True)
                                      for w_train, w_target in zip(train_weights, target_weights)]

    def _build_network(self):
        """
        Build the model's network. (Example implementation in doom_rl.models.tfmodels.SimpleTfModel)

        To implement this method you should:
            Build the structure of your network. Use self.s_input as your network's input and return
            the output of the network. The number of units in the output layer is self.nb_actions.

        Returns:
            The output of the network.
        """

        raise NotImplementedError()

    @property
    def session(self):
        if not self._model_created:
            raise RuntimeError('A model must be compiled before performing any operations.')
        if self._session is None:
            self._session = tf.Session()
            self._session.run(tf.global_variables_initializer())
        return self._session


# class DqnKerasModel(Model):
#     """
#     This is a half-implemented DQN model based on tensorflow.keras module. All abstract methods defined
#     in base class `Model` are implemented.
#     You can easily extend this class by just implementing `_build_network()` method in your subclass.
#
#     Args:
#         state_shape: The shape of input states, which indicates the input shape of this model's
#         input layer.
#         nb_actions: The number of actions that the agent can perform, which indicates the number of
#         units in this model's output layer.
#     """
#     def __init__(self, state_shape, nb_actions, **kwargs):
#         super(DqnKerasModel, self).__init__(**kwargs)
#         self.state_shape = state_shape
#         self.nb_actions = nb_actions
#
#         # Keras model
#         self._model = None
#
#         # Keras input layer and output layer
#         self.input_layer = tfk.layers.Input(shape=self.state_shape, dtype="float32", name="InputLayer")
#         self.output_layer = None
#
#         # Tensorflow session
#         self._session = None
#
#         # State input of the network
#         self.s_input = None
#
#         # Action input, indicating the actions chosen under self.s_input
#         self.a_input = None
#
#         # Output q values, must be used in _build_network() method as the output of the network
#         self.q_values = None
#
#         # Optimizer
#         self._optimizer = None
#
#         # Operation nodes defined by _build_network() method
#         self._max_q_values = None
#         self._best_action = None
#         self._action_q_values = None
#         self._loss = None
#         self._train = None
#
#         # summary
#         # ...
#
#         self._model_created = False
#
#     def save_weights(self, save_path):
#         self._model.save_weights(save_path)
#
#     def load_weights(self, load_path):
#         self._model.load_weights(load_path)
#
#     def train(self, state, action, target_q):
#         state = self.process_state_batch(state)
#         l, _ = self.session.run([self._loss, self._train],
#                                 {self.s_input: state, self.target_q_values: target_q, self.a_input: action})
#         return l
#
#     def get_best_action(self, state):
#         state = self.process_state_batch(state)
#         return self.session.run(self._best_action, {self.s_input: state})[0]
#
#     def get_q_values(self, state):
#         state = self.process_state_batch(state)
#         return self.session.run(self.q_values, {self.s_input: state})
#
#     def get_max_q_values(self, state):
#         state = self.process_state_batch(state)
#         return self.session.run(self._max_q_values, {self.s_input: state})
#
#     def compile(self, learning_rate, optimizer='Adam', **kwargs):
#         optimizer = optimizer.lower()
#         optimizers = {'adam': tf.train.AdamOptimizer,
#                       'rmsprop': tf.train.RMSPropOptimizer,
#                       'sgd': tf.train.GradientDescentOptimizer}
#         if optimizer not in optimizers:
#             raise KeyError('Invalid optimizer {}.'.format(optimizer))
#
#         if not self._model_created:
#             self._create_placeholders()
#             self._build_network()
#             # Check whether the output layer is defined
#             assert self.output_layer is not None, 'The output of the network is not defined.'
#
#             self._create_operations()
#             self._model_created = True
#
#         self._optimizer = optimizers[optimizer](learning_rate=learning_rate, **kwargs)
#         self._train = self._optimizer.minimize(self._loss)
#
#     def _build_network(self):
#         """
#         Build the model's network. (Example implementation in doom_rl.models.kmodels.SimpleKerasModel)
#
#         To implement this method you should:
#             Build the structure of your network. Use self.input_layer as your network's input and self.output_layer
#             as your network's output. The number of units in the output layer is self.nb_actions.
#         """
#         raise NotImplementedError()
#
#     def _create_placeholders(self):
#         assert not self._model_created
#
#         self.s_input = tf.placeholder(tf.float32, shape=[None] + list(self.state_shape), name='InputState')
#         self.a_input = tf.placeholder(tf.int32, shape=[None], name='InputAction')
#         self.target_q_values = tf.placeholder(tf.float32, shape=[None], name='OutputTargetQ')
#
#     def _create_operations(self):
#         assert not self._model_created
#
#         self._model = tfk.models.Model(inputs=self.input_layer, outputs=self.output_layer)
#         self.q_values = self._model(self.s_input)
#         self._max_q_values = tf.reduce_max(self.q_values, axis=1)
#         self._best_action = tf.argmax(self.q_values, axis=1)
#         self._action_q_values = tf.reduce_sum(self.q_values * tf.one_hot(self.a_input, self.nb_actions), axis=1)
#         self._loss = tf.losses.mean_squared_error(self._action_q_values, self.target_q_values)
#
#     @property
#     def session(self):
#         if not self._model_created:
#             raise RuntimeError('A model must be compiled before performing any operations.')
#         if self._session is None:
#             self._session = tf.Session()
#             self._session.run(tf.global_variables_initializer())
#         return self._session
