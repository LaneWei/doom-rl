from doom_rl.memory import Memory
from doom_rl.models.model import Model


class Agent:
    """
    Base class for implementing different agents.

    You need to implement the following method:
        * `learn_from_memory`

    Args:
        model: A compiled nn model. (`Model` instance).
        memory: Agent's memory (`Memory` instance).
        actions: A list of actions that this agent can take.
        # policy: ...
        # learning_rate: The learning rate that will be applied to the learning process.
        discount_factor: The discount factor that will be applied to the learning process.
    """

    def __init__(self, model, memory, actions, discount_factor=0.95):
        self._model = model
        self._memory = memory
        self._actions = actions
        # self._lr = None
        self._gamma = None

        # self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, state):
        """
        Get the action that this agent should perform at the current state `state`.

        Args:
            state: The current state.

        Returns:
            The action according to this agent's model.

        """

        return self.model.get_best_action(state)

    def learn_from_memory(self, batch_size):
        """
        Perform a learning step. This agent will get a randomly chosen batch (see doom_rl.memory.Memory.sample for
        more information about how the batch is chosen) of experiences from its memory and learn to train its model.

        Args:
            batch_size: The expected size of the experience batch which will be used for agent's learning step.

        Returns:
            The loss of this learning step.
        """
        pass

    def save_experience(self, s, a, r, s_, terminate):
        """
        See `memory.add` for more information.

        Args:
            s: Current state.
            a: Action.
            r: Reward.
            s_: Next state.
            terminate: The terminate flag.

        Returns:
            True, if the experience is saved.
        """

        return self.memory.add(s, a, r, s_, terminate)

    @property
    def model(self):
        if not isinstance(self._model, Model):
            raise ValueError('The model should be an instance of doom_rl.models.model.Model')
        return self._model

    @property
    def memory(self):
        if not isinstance(self._memory, Memory):
            raise ValueError('The memory should be an instance of doom_rl.memory.Memory')
        return self._memory

    @property
    def actions(self):
        return list(self._actions)

    @property
    def action_size(self):
        return len(self._actions)

    """
    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError('The value of learning rate should always be positive.')
        self._lr = value
    """

    @property
    def discount_factor(self):
        return self._gamma

    @discount_factor.setter
    def discount_factor(self, value):
        if value < 0:
            raise ValueError('The value of discount factor should always be non-negative.')
        self._gamma = value
