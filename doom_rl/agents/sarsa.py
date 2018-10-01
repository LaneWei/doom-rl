from doom_rl.agents.agent import Agent
import numpy as np


class ESARSAAgent(Agent):
    """
        sarsa algorithm implementation
    """
    def __init__(self, model, memory, actions, **kwargs):
        super(ESARSAAgent, self).__init__(model, memory, actions, **kwargs)

    def learn_from_memory(self, batch_size, policy=None):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)

        distribution = policy.action_probs()
        q_values_next_state = self.model.get_q_values(next_states)
        assert distribution.ndim == 1
        assert distribution.shape == q_values_next_state.shape

        q_next_state = np.sum(q_values_next_state * distribution)
        target_q = rewards + q_next_state * self.discount_factor * (1 - terminates)
        return self.model.train(states, actions, target_q)
