from doom_rl.utils.core.agent import Agent
import numpy as np


class DQNAgent(Agent):
    """
        Agent implemented with q-learning algorithm.
    """
    def __init__(self, model, memory, actions, **kwargs):
        super(DQNAgent, self).__init__(model, memory, actions, **kwargs)

    def learn_from_memory(self, batch_size):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)

        target_q = rewards + self.model.get_max_q_values(next_states) * self.discount_factor * (1 - terminates)
        return self.model.train(states, actions, target_q)


class ESARSAAgent(Agent):
    """
    Agent implemented with expected sarsa algorithm.
    """
    def __init__(self, model, memory, actions, epsilon=0, **kwargs):
        super(ESARSAAgent, self).__init__(model, memory, actions, **kwargs)
        self.epsilon = epsilon

    def learn_from_memory(self, batch_size):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)

        eps = self.epsilon
        q_values_next_state = self.model.get_q_values(next_states)
        q_next_state = np.mean(q_values_next_state, axis=1) * eps + np.max(q_values_next_state, axis=1) * (1 - eps)

        target_q = rewards + q_next_state * self.discount_factor * (1 - terminates)
        return self.model.train(states, actions, target_q)
