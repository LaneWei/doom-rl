import numpy as np


class Policy:
    """
    Abstract base class for q values based policies.

    Note: The input q_values can either be a list of q values or a batch (lists) of q values.

    You need to implement the following method:
        * `action_probs`
    """

    def action_probs(self, q_values):
        """
        Get the probability distribution, determined by this policy, of a list of actions given their q values.

        Args:
            q_values: An numpy.ndarray, whose ndim is 1 or 2, that contains the q values of all available actions.

        Returns:
            An numpy.ndarray containing the probability distribution with the same shape of q_values.
        """
        raise NotImplementedError()

    def choose_action(self, q_values):
        """
        Choose an action according to some policy:
            - write me

        Args:
            q_values: An numpy.ndarray, whose ndim is 1 or 2, that contains the q values of all available actions.

        Returns:
            An integer or an numpy.ndarray, whose ndim is (q_values.ndim - 1), of the chosen action(s).
        """

        assert q_values.ndim == 1 or q_values.ndim == 2

        action_probs = self.action_probs(q_values)
        if q_values.ndim == 1:
            return int(np.random.choice(range(len(q_values)), p=action_probs))

        actions = [np.random.choice(range(len(probs)), p=probs) for probs in action_probs]
        return np.array(actions, dtype=np.int32)


class GreedyPolicy(Policy):
    """
    Greedy policy.
    This policy chooses the action with the highest q value.
    """

    def action_probs(self, q_values):
        if q_values.ndim == 1:
            q_values = q_values.reshape(1, q_values.size)
        assert q_values.ndim == 2

        arg_max_q = np.argmax(q_values, axis=1)
        probs = np.zeros_like(q_values, dtype=np.float32)
        probs[np.arange(len(arg_max_q)), arg_max_q] = 1
        return probs.reshape(q_values.shape)

    def choose_action(self, q_values):
        """
        Choose an action according to greedy policy:
            - The action with the highest q value will be chosen.

        Args:
            q_values: An numpy.ndarray, whose ndim is 1 or 2, that contains the q values of all available actions.

        Returns:
            An numpy.ndarray, whose ndim is (q_values.ndim - 1), of the chosen action(s).
        """

        return super(GreedyPolicy, self).choose_action(q_values)


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

    def action_probs(self, q_values):
        if q_values.ndim == 1:
            q_values = q_values.reshape(1, q_values.shape[0])
        assert q_values.ndim == 2

        probs = np.ones_like(q_values, dtype=np.float32) * (self.epsilon / q_values.shape[1])
        arg_max_q = np.argmax(q_values, axis=1)
        probs[np.arange(len(arg_max_q)), arg_max_q] += 1 - self.epsilon
        return probs.reshape(q_values.shape)

    def choose_action(self, q_values):
        """
        Choose an action according to epsilon greedy policy:
            - The action with the highest q value has a (1 - eps) probability of being chosen.
            - Random action is to be chosen with probability eps.

       Args:
            q_values: An numpy.ndarray, whose ndim is 1 or 2, that contains the q values of all available actions.

        Returns:
            An numpy.ndarray, whose ndim is (q_values.ndim - 1), of the chosen action(s).
        """

        return super(EpsilonGreedyPolicy, self).choose_action(q_values)

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
