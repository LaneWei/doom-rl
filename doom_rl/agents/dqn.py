from doom_rl.agents.agent import Agent


class DQNAgent(Agent):
    """
        q-learning algorithm implementation
    """
    def __init__(self, model, memory, actions, **kwargs):
        super(DQNAgent, self).__init__(model, memory, actions, **kwargs)

    def learn_from_memory(self, batch_size, policy=None):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)

        target_q = rewards + self.model.get_max_q_values(next_states) * self.discount_factor * (1 - terminates)
        return self.model.train(states, actions, target_q)
