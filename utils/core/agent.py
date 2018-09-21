
class Agent:
    """
    Base class for implementing different agents.

    # Argument
        * `create_model` A function that takes no parameter and return a model (`Model` instance).
    """

    def __init__(self, create_model):
        self.Model = create_model()

        self._lr = 1e-5
        self._gamma = 0.95

    def get_action(self, state):
        """
        Get the action that this agent should perform at current state `state` according to the agent's model.

        :param state: The current state.
        :return: The action according to this agent's model.
        """
        pass

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
        pass

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
