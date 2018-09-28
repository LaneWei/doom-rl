from doom_rl.memory import Memory
from doom_rl.models.model import Model


class Agent:
    """
    Base class for implementing different agents.

    You need to implement the following method:
        * `learn_from_memory`

    # Argument
        * `model` An nn model. (`Model` instance).
        * `memory` Agent's memory (`Memory` instance).
        * `actions' A set or list of actions that this agent can take.
        * `policy` ...
        * `learning_rate` The learning rate that will be applied to the learning process.
        * `discount_factor` The discount factor that will be applied to the learning process.
    """

    def __init__(self, model, memory, actions, learning_rate=1e-5, discount_factor=0.95):
        self._model = model
        self._memory = memory
        self._actions = actions
        self._lr = None
        self._gamma = None

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, state):
        """
        Get the action that this agent should perform at current state `state` according to the agent's model.

        :param state: The current state.
        :return: The action according to this agent's model.
        """

        return self.model.get_best_action(state)

    def learn_from_memory(self, batch_size):
        """
        Perform the learning step. This agent will get a randomly chosen batch of experiences from its
        memory and learn to improve its model.

        :param batch_size: The expected size of batch.
        """
        pass

    def save_experience(self, s, a, r, s_, terminate):
        """
        See `memory.add` for more information.

        :param s: Current state.
        :param a: Action.
        :param r: Reward.
        :param s_: The next state.
        :param terminate: The terminate flag.
        :return: True, if the experience is saved to this agent's memory successfully.
        """

        return self.memory.add(s, a, r, s_, terminate)

    @property
    def model(self):
        if not isinstance(self._model, Model):
            raise ValueError('The model should be an instance of utils.core.model.Model')
        return self._model

    @property
    def memory(self):
        if not isinstance(self._memory, Memory):
            raise ValueError('The memory should be an instance of utils.core.memory.Memory')
        return self._memory

    @property
    def actions(self):
        return self._actions

    @property
    def action_size(self):
        return len(self._actions)

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError('The value of learning rate should always be positive.')
        self._lr = value

    @property
    def discount_factor(self):
        return self._gamma

    @discount_factor.setter
    def discount_factor(self, value):
        if value < 0 or value > 1:
            raise ValueError('The value of discount factor should always within the range of [0, 1].')
        self._gamma = value
