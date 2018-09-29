import numpy as np


class Policy:
    """
    Abstract base class for q values based policies.

    You need to implement the following method:
        * `choose_action`
    """

    def choose_action(self, q_values):
        pass


class GreedyPolicy(Policy):
    """
    Greedy policy.
    This policy chooses the action with the highest q value.
    """

    def choose_action(self, q_values):
        assert q_values.ndim == 1

        return np.argmax(q_values)


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon greedy policy with epsilon decay.
    Epsilon greedy policy chooses an action randomly with probability eps, or chooses the action with
    the highest q value with probability 1 - eps.

    Args:
        start_epsilon: The floating point value ([end_epsilon, 1]) of epsilon that will be taken at the first step.
        end_epsilon: The floating point value ([0, start_epsilon]) of epsilon that will be taken after
        steps reaches total_decay_steps.
        total_decay_steps: The number of steps to take the value of epsilon from start_epsilon to end_epsilon.
    """

    def __init__(self, start_epsilon=0., end_epsilon=0., total_decay_steps=1):
        super(EpsilonGreedyPolicy, self).__init__()
        self._start_epsilon = start_epsilon if start_epsilon <= 1 else 1
        self._end_epsilon = end_epsilon if end_epsilon >= 0 else 0
        self._decay_steps = total_decay_steps if total_decay_steps > 0 else 1
        self._steps = 0

        if self._end_epsilon > self._start_epsilon:
            self._end_epsilon = self._start_epsilon

    def choose_action(self, q_values):
        """
        Choose an action according to epsilon greedy policy.

        Args:
            q_values: An numpy.ndarray that contains the q values of all available actions.

        Returns:
            An action (an integer).
        """

        assert q_values.ndim == 1

        if np.random.uniform() < self.epsilon:
            action = np.random.choice(q_values.shape[0])
        else:
            action = np.argmax(q_values)
        return action

    def update(self, steps):
        """
        Update the number of training steps that the agent has performed.

        Args:
            steps: The number of training steps the agent has performed.
        """

        if steps < 0:
            self._steps = 0
        elif steps > self._decay_steps:
            self._steps = self._decay_steps
        else:
            self._steps = steps

    @property
    def epsilon(self):
        return self._start_epsilon + float(self._steps / self._decay_steps) * (self._end_epsilon - self._start_epsilon)
