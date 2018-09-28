from random import random


class EpsilonGreedyPolicy:
    """
    Epsilon greedy policy with epsilon decay.

    Args:
        start_epsilon: The value ([end_epsilon, 1]) of epsilon that will be taken at the first step.
        end_epsilon: The value ([0, start_epsilon]) of epsilon that will be taken after steps reaches decay_steps.
        decay_steps: The number of steps to take the value of epsilon from start_epsilon to end_epsilon.
    """

    def __init__(self, start_epsilon=0., end_epsilon=0., decay_steps=1):
        self._start_epsilon = start_epsilon if start_epsilon <= 1 else 1
        self._end_epsilon = end_epsilon if end_epsilon >= 0 else 0
        self._decay_steps = decay_steps if decay_steps > 0 else 1
        self._steps = 0

        if self._end_epsilon > self._start_epsilon:
            self._end_epsilon = self._start_epsilon

    def choose_by_random(self):
        """
        Return whether to choose action by random according to epsilon greedy policy.
        """

        if random() < self.epsilon:
            return True
        return False

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
