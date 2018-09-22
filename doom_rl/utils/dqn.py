from doom_rl.utils.core.agent import Agent


class DQNAgent(Agent):
    """
        Agent implemented with q-learning algorithm.
    """
    def __init__(self, create_model, memory, **kwargs):
        super(DQNAgent, self).__init__(create_model, memory, **kwargs)

    def learn_from_memory(self, batch_size):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)

        target_q = rewards + self.model.get_max_q_values(next_states) * self.discount_factor * (1 - terminates)
        return self.model.train(states, actions, target_q)
