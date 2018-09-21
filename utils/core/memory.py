from collections import namedtuple
from random import sample

# An agent's experience is a tuple representing the transition of `current state`, `taken action`,
# `received reward`, `next state`, and `terminate flag`.
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'terminate'])


class Memory:
    """
    An abstract base class for storing agent's experience.

    You need to implement the following two methods:
        * `add`
        * `sample`

    # Arguments
        * `state_shape` A tuple that specifies the shape of states which are stored in this memory.
        * `capacity` A non-negative integer that specifies the maximum size of this memory.

    # Property
        * `size` An integer which indicates how many 'experiences' are currently stored in this memory.
    """

    def __init__(self, state_shape, capacity=10000):
        self.state_shape = state_shape
        self.capacity = capacity

    def add(self, s, a, r, s_, terminate):
        """
        Add an experience to this memory.
        If the size of this memory reaches its capacity, the 'experience' to be added will take the
        place of one of the former (oldest or whatever) 'experience' in this memory.

        :param s: Current state. If the shape of s can not match self.state_shape, this method should
        fail in some way (return False, throwing an error, etc.)
        :param a: Taken action.
        :param r: Received reward.
        :param s_: Next state. If the shape of s can not match self.state_shape, this method should
        fail in some way (return False, throwing an error, etc.)
        :param terminate: Terminate flag.
        :return: True if s, a, r, s_, terminate are saved as an experience and added to this memory
        """
        pass

    def sample(self, batch_size):
        """
        Returns randomly selected `experience` batch.

        :param batch_size: The expected size of `experience` batch.
        :return: A tuple of five lists, containing elements of s, a, r, s_, terminate respectively,
        in which each of the lists has a size of max(self.size, batch_size).
        """
        pass


class ListMemory(Memory):
    def __init__(self, state_shape, **kwargs):
        super(ListMemory, self).__init__(state_shape, **kwargs)

        self._storage = []
        self._index = 0

    def add(self, s, a, r, s_, done):
        exp = Experience(s, a, r, s_, done)

        if self.size < self.capacity:
            self._storage.append(exp)
        else:
            self._storage[self._index] = exp
        self._index = (self._index + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > self.size:
            samples = self._storage
        else:
            samples = sample(self._storage, batch_size)

        states = [x.state for x in samples]
        actions = [x.action for x in samples]
        rewards = [x.reward for x in samples]
        next_states = [x.next_state for x in samples]
        d = [x.terminate for x in samples]

        return states, actions, rewards, next_states, d

    @property
    def size(self):
        return len(self._storage)