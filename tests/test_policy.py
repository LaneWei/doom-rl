import pytest

from collections import defaultdict
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from doom_rl.policy import GreedyPolicy, EpsilonGreedyPolicy

np.random.seed(1)


def get_random_q_values(high, shape):
    return np.random.uniform(high, size=shape)


class TestGreedyPolicy:
    def test_qvalues_ndim_1(self):
        policy = GreedyPolicy()

        q_values1 = np.array([1.0, 2.0, 2.5, 5.5, 3.0, 1.5], dtype=np.float32)
        action_probs1 = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        action1 = 3
        assert_allclose(policy.action_probs(q_values1), action_probs1)
        assert policy.choose_action(q_values1) == action1

        q_values2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        action_probs2 = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
        action2 = 0
        assert_allclose(policy.action_probs(q_values2), action_probs2)
        assert policy.choose_action(q_values2) == action2

        q_values_random = get_random_q_values(high=100, shape=100)
        action_probs_random = np.zeros([100], dtype=np.float32)
        action_random = np.argmax(q_values_random)
        action_probs_random[action_random] = 1.
        assert_allclose(policy.action_probs(q_values_random), action_probs_random)
        assert policy.choose_action(q_values_random) == action_random

    def test_qbalues_ndim_2(self):
        policy = GreedyPolicy()

        q_values1 = np.array([[1.0, 2.0, 2.5, 5.5, 3.0, 1.5],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 2.0, 2.5, 5.5, 3.0, 8.5]], dtype=np.float32)
        action_probs1 = np.array([[0, 0, 0, 1, 0, 0],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        action1 = np.array([3, 0, 5], dtype=np.int32)
        assert_allclose(policy.action_probs(q_values1), action_probs1)
        assert_equal(policy.choose_action(q_values1), action1)

        q_values_random = get_random_q_values(high=100, shape=(100, 100))
        action_random = np.argmax(q_values_random, axis=1)
        action_probs_random = np.eye(100, dtype=np.int32)[action_random]
        assert_allclose(policy.action_probs(q_values_random), action_probs_random)
        assert_equal(policy.choose_action(q_values_random), action_random)


class TestEpsilonGreedyPolicy:
    def test_epsilon(self):
        policy = EpsilonGreedyPolicy()
        assert abs(policy.epsilon) < 1e-6
        policy.update(1000)
        assert abs(policy.epsilon) < 1e-6

        decay_steps = 0
        policy1 = EpsilonGreedyPolicy(start_epsilon=0.8, end_epsilon=0.0, total_decay_steps=decay_steps)
        assert abs(policy1.epsilon) < 1e-6

        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(start_epsilon=0.8, end_epsilon=0.0, total_decay_steps=-1)
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(start_epsilon=0.8, end_epsilon=-1)
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(start_epsilon=1.5)
        with pytest.raises(ValueError):
            EpsilonGreedyPolicy(start_epsilon=0.0, end_epsilon=0.8)

        decay_steps = 10
        policy2 = EpsilonGreedyPolicy(start_epsilon=1.0, end_epsilon=0.5, total_decay_steps=decay_steps)
        epsilons = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        for i in range(decay_steps * 2):
            epsilon = policy2.epsilon
            assert abs(epsilon - epsilons[i if i < decay_steps else decay_steps]) < 1e-6
            policy2.update(i + 1)

    def test_qvalues_ndim_1(self):
        policy1 = EpsilonGreedyPolicy(start_epsilon=0.35)
        policy2 = EpsilonGreedyPolicy(start_epsilon=np.random.random())
        policy2_epsilon = policy2.epsilon
        action_repeat = 10000
        tolerance = 0.02

        q_values1 = np.array([1.0, 2.0, 2.5, 5.5, 3.0], dtype=np.float32)
        action_probs1_p1 = np.array([0.07, 0.07, 0.07, 0.72, 0.07], dtype=np.float32)
        action_probs1_p2 = np.array([policy2_epsilon / q_values1.size for _ in range(q_values1.size)])
        action_probs1_p2[3] += 1 - policy2_epsilon
        assert_allclose(policy1.action_probs(q_values1), action_probs1_p1)
        assert_allclose(policy2.action_probs(q_values1), action_probs1_p2)
        # Monte Carlo method
        actions1_p1 = defaultdict(lambda: 0)
        actions1_p2 = defaultdict(lambda: 0)
        for _ in range(action_repeat):
            actions1_p1[policy1.choose_action(q_values1)] += 1
            actions1_p2[policy2.choose_action(q_values1)] += 1
        actions1_distribution_p1 = [actions1_p1[i] / action_repeat for i in range(len(q_values1))]
        actions1_distribution_p2 = [actions1_p2[i] / action_repeat for i in range(len(q_values1))]
        assert_allclose(actions1_distribution_p1, action_probs1_p1, atol=tolerance)
        assert_allclose(actions1_distribution_p2, action_probs1_p2, atol=tolerance)

        q_values_random = get_random_q_values(high=100, shape=10)
        action_probs_random_p2 = np.array([policy2_epsilon / q_values_random.size for _ in range(q_values_random.size)])
        action_probs_random_p2[np.argmax(q_values_random)] += 1 - policy2_epsilon
        assert_allclose(policy2.action_probs(q_values_random), action_probs_random_p2)
        # Monte Carlo method
        actions_random_p2 = defaultdict(lambda: 0)
        for _ in range(action_repeat):
            actions_random_p2[policy2.choose_action(q_values_random)] += 1
        actions_random_distribution_p2 = [actions_random_p2[i] / action_repeat for i in range(len(q_values_random))]
        assert_allclose(actions_random_distribution_p2, action_probs_random_p2, atol=tolerance)

    def test_qbalues_ndim_2(self):
        policy = EpsilonGreedyPolicy(start_epsilon=0.35)

        q_values1 = np.array([[1.0, 2.0, 2.5, 5.5, 3.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 2.0, 2.5, 5.5, 8.5]], dtype=np.float32)
        action_probs1 = np.array([[0.07, 0.07, 0.07, 0.72, 0.07],
                                  [0.72, 0.07, 0.07, 0.07, 0.07],
                                  [0.07, 0.07, 0.07, 0.07, 0.72]], dtype=np.float32)

        qvalues_shape = (100, 100)
        q_values_random = get_random_q_values(high=100, shape=qvalues_shape)
        action_probs_random = np.ones(qvalues_shape) * (policy.epsilon / q_values_random.shape[-1])
        action_probs_random += np.eye(q_values_random.shape[-1])[np.argmax(q_values_random, axis=1)]*(1-policy.epsilon)

        assert_allclose(policy.action_probs(q_values1), action_probs1)
        assert_allclose(policy.action_probs(q_values_random), action_probs_random)


if __name__ == "__main":
    pytest.main(__file__)
