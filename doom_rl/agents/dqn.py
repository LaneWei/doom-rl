from doom_rl.agents.agent import Agent
from doom_rl.utils import process_batch


class DQNAgent(Agent):
    """
        q-learning algorithm implementation
    """
    def __init__(self, model, memory, actions):
        super(DQNAgent, self).__init__(model, memory, actions)

    def learn_from_memory(self, batch_size, policy=None):
        states, actions, rewards, next_states, terminates = self.memory.sample(batch_size)
        states = process_batch(states)
        next_states = process_batch(next_states)

        return self.model.train(states, actions, rewards, next_states, terminates)
