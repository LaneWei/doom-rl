import pytest

import numpy as np
from numpy.testing import assert_allclose
import warnings
from doom_rl.memory import Experience
from doom_rl.memory import ListMemory

np.random.seed(1)
state_shape = (32, 32)


def get_random_experience(shape):
    state = np.random.random(shape)
    action = np.random.random()
    reward = np.random.random()
    state_ = np.random.random(shape)
    done = np.random.uniform() < 0.5
    return Experience(state, action, reward, state_, done)


class TestListMemory:
    def test_add_fail(self):
        wrong_shape = (30, 30)

        memory = ListMemory(1, state_shape=state_shape)
        experience = get_random_experience(wrong_shape)

        warnings.filterwarnings("ignore")
        assert not memory.add(*experience)
        assert memory.size == 0

    def test_add_max_capacity(self):
        MAX = 3

        memory = ListMemory(MAX, state_shape=state_shape)
        experiences = [get_random_experience(state_shape) for _ in range(MAX + 1)]
        first_experience = experiences[0]

        for i in range(len(experiences)):
            assert memory.add(*experiences[i])
            assert memory.size == (i + 1) if i < MAX else MAX

        # No first experience
        for state, _, _, state_, _ in zip(*memory.sample(MAX)):
            assert not np.all(np.abs(first_experience.state - state) < 1e-6)
            assert not np.all(np.abs(first_experience.next_state - state_) < 1e-6)

    def test_sample(self):
        MAX = 100
        test_batch_size = 16
        small_batch_size = 3

        memory = ListMemory(MAX, state_shape=state_shape)
        experiences = [get_random_experience(state_shape) for _ in range(small_batch_size)]
        for experience in experiences:
            memory.add(*experience)

        states, _, _, states_, _ = memory.sample(test_batch_size)
        assert states.shape == (small_batch_size,) + state_shape
        assert states_.shape == (small_batch_size,) + state_shape
        for i in range(small_batch_size):
            assert_allclose(experiences[i].state, states[i])
            assert_allclose(experiences[i].next_state, states_[i])

        experiences = [get_random_experience(state_shape) for _ in range(MAX)]
        for experience in experiences:
            memory.add(*experience)

        states, _, _, states_, _ = memory.sample(test_batch_size)
        assert states.shape == (test_batch_size,) + state_shape
        assert states_.shape == (test_batch_size,) + state_shape


if __name__ == "__main":
    pytest.main(__file__)
