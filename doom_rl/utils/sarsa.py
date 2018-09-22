from doom_rl.utils.core.agent import Agent
from doom_rl.utils.policy import EpsilonGreedyPolicy
import numpy as np


class ESARSA(Agent):
    """
    Agent implemented with expected sarsa algorithm.
    """
    def __init__(self, create_model, memory, policy=EpsilonGreedyPolicy(), **kwargs):
        super(ESARSA, self).__init__(create_model, memory, **kwargs)
        self.policy = policy

    def learn_from_memory(self, batch_size):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)

        eps = self.policy.epsilon
        q_values_next_state = self.model.get_q_values(next_states)
        q_next_state = np.mean(q_values_next_state, axis=1) * eps + np.max(q_values_next_state, axis=1) * (1 - eps)

        target_q = rewards + q_next_state * self.discount_factor * (1 - terminates)
        return self.model.train(states, actions, target_q)
