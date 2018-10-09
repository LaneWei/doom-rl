from doom_rl.agents.agent import Agent
import numpy as np


# class ESarsaAgent(Agent):
#     """
#         expected sarsa algorithm implementation
#
#         This agent's `learn_from_memory()` requires a policy to be provided.
#     """
#     def __init__(self, model, memory, actions, **kwargs):
#         super(ESarsaAgent, self).__init__(model, memory, actions, **kwargs)
#
#     def learn_from_memory(self, batch_size, policy=None):
#         states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)
#
#         q_values_next_state = self.model.get_q_values(next_states)
#         distribution = policy.action_probs(q_values_next_state)
#         assert distribution.shape == q_values_next_state.shape
#
#         q_next_state = np.sum(q_values_next_state * distribution, axis=1)
#         target_q = rewards + q_next_state * self.discount_factor * (1 - terminates)
#         return self.model.train(states, actions, target_q)
