from doom_rl.memory import Memory
from doom_rl.models.model import Model
from doom_rl.policy import GreedyPolicy
import numpy as np


class Agent:
    """
    Base class for implementing different agents.

    You need to implement the following method:
        * `learn_from_memory`

    Args:
        model: A compiled nn model. (`Model` instance).
        memory: Agent's memory (`Memory` instance).
        actions: A list of actions that this agent can take. (May be different from env.action_space)
        discount_factor: The discount factor that will be applied to the learning process.
    """

    def __init__(self, model, memory, actions, discount_factor=0.95):
        self._model = model
        self._memory = memory
        self._action_space = actions
        # self._lr = None
        self._gamma = None

        # self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_q_values(self, state):
        """
        Get the q values of all available actions at a given state.

        Args:
            state: The current state.

        Returns:
            An numpy.ndarray, containing the q values.
        """

        return np.asarray(self.model.get_q_values([state])[0], dtype=np.float32)

    def get_action(self, state, policy=GreedyPolicy()):
        """
        Get the action that this agent should perform at the current state `state` according to
        the given policy. If the policy is not provided, greedy policy (choose the action with
        the highest q value) will be applied

        Args:
            state: The current state.
            policy: A policy. (see doom_rl.policy for more information)

        Returns:
            The action chosen by the given policy.
        """

        action = policy.choose_action(q_values=self.get_q_values(state))
        return action

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
    def action_space(self):
        return list(self._action_space)

    @property
    def discount_factor(self):
        return self._gamma

    @discount_factor.setter
    def discount_factor(self, value):
        if value < 0:
            raise ValueError('The value of discount factor should always be non-negative.')
        self._gamma = value
