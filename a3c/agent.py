import threading
import numpy as np
from collections import deque
from time import sleep
from tqdm import trange

from config import Config
from utils import ThreadDelay


class EpsilonGreedyPolicy:
    def __init__(self, start_epsilon=0., end_epsilon=0., total_decay_steps=1):
        self._start_epsilon = start_epsilon
        self._end_epsilon = end_epsilon
        self._decay_steps = total_decay_steps
        self.steps = 0

        if self._end_epsilon > self._start_epsilon:
            self._end_epsilon = self._start_epsilon

    def choose_action(self, probs):
        epsilon = self.get_epsilon()
        self.steps += 1
        size = len(probs)
        if np.random.rand() < epsilon:
            probs = np.ones(size, np.float32) / size
        return int(np.random.choice(size, p=probs))

    def get_epsilon(self):
        if self.steps >= self._decay_steps:
            self.steps = self._decay_steps
            return self._end_epsilon
        return self._start_epsilon + float(self.steps / self._decay_steps) * (self._end_epsilon - self._start_epsilon)


class A3CAgent:
    def __init__(self, network, game_env, policy=None, run_episodes=1000000):
        self.network = network
        self.game_env = game_env
        self.policy = policy
        self.run_episodes = run_episodes

        self.n_step_return = Config.N_STEP_RETURN
        self.frame_repeat = Config.FRAME_REPEAT
        self.gamma = Config.GAMMA
        self.gamma_n = self.gamma ** self.n_step_return

        self.action_size = len(Config.ACTION_SPACE)
        self.i2a = {index: action for index, action in enumerate(np.eye(self.action_size))}
        self.R = 0

        self.episode_rewards = []

    def get_v_and_pi(self, state):
        v, pi = self.network.predict([state])
        return v[0], pi[0]

    def act(self, s):
        v, pi = self.get_v_and_pi(s)
        if self.policy:
            action_index = self.policy.choose_action(pi)
        else:
            action_index = np.argmax(pi)
        return v, pi, action_index

    def clear_rewards(self):
        rewards = self.episode_rewards
        self.episode_rewards = []
        return rewards


class A3CTrainAgent(A3CAgent, threading.Thread):
    def __init__(self, network, game_env, policy=None, run_episodes=1000000):
        A3CAgent.__init__(self, network, game_env, policy, run_episodes)
        threading.Thread.__init__(self)

        self.memory = deque(maxlen=self.n_step_return)
        self.episode_rewards = []
        self.RUN = True
        self.thread_delay = ThreadDelay(1.5e-3)

    def save_experience(self, state, action, reward, state_, mask):
        def get_sample(n):
            s, a, _, _, _ = self.memory[0]
            _, _, r, s_, m = self.memory[n - 1]

            return s, a, r, s_, m
        self.memory.append((state, action, reward, state_, mask))

        self.R = (self.R + reward * self.gamma_n) / self.gamma

        if mask == 0:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, _, s_, m = get_sample(n)
                self.network.add_sample(s, a, self.R, s_, m)
                first_r = self.memory.popleft()[2]
                self.R = (self.R - first_r) / self.gamma

            self.R = 0

        if len(self.memory) == self.n_step_return:
            s, a, _, s_, m = get_sample(self.n_step_return)
            status = self.network.add_sample(s, a, self.R, s_, m)
            self.thread_delay.delay_on_fail(status)
            first_r = self.memory[0][2]
            self.R = self.R - first_r

    def stop(self):
        self.RUN = False

    def run(self):
        for episode in range(self.run_episodes):
            if not self.RUN:
                break

            s = self.game_env.reset()
            steps = 0
            terminate = False
            while self.RUN and not terminate:
                # choose an action
                v, pi, a_index = self.act(s)
                # take one step
                s_, r, terminate = self.game_env.step(a_index, self.frame_repeat, reward_discount=100)
                # save
                self.save_experience(s, self.i2a[a_index], r, s_, 0 if terminate else 1)
                # Update the current state
                s = s_
                steps += 1
                if terminate:
                    # Record the total amount of reward in this episode
                    reward = self.game_env.episode_reward()
                    self.episode_rewards.append(reward)


class A3CTestAgent(A3CAgent):
    def __init__(self, network, game_env, policy=None,
                 run_episodes=1, verbose=False, spectator_mode=False):
        A3CAgent.__init__(self, network, game_env, policy, run_episodes)
        self.VERBOSE = verbose
        self.SPECTATOR_MODE = spectator_mode

    def run(self):
        death = 0
        for episode in trange(self.run_episodes):
            if self.VERBOSE:
                if self.SPECTATOR_MODE:
                    sleep(1.5)
                print('\nEpisode %d:' % (episode + 1))

            s = self.game_env.reset()
            terminate = False
            while not terminate:
                v, pi, a_index = self.act(s)
                if self.VERBOSE:
                    print('V: %6.3f' % v, end='   ')
                    print('Pi: [', end=' ')
                    for value in pi:
                        print('%.3f' % value, end=' ')
                    print(']', end='\t')
                    print('action: ', a_index + 1, ', ', self.game_env.get_action(a_index), end='\t')

                    s_, r, terminate = self.game_env.step(a_index, self.frame_repeat, slow_update=True)
                    print('reward: ', r)
                else:
                    s_, r, terminate = self.game_env.step(a_index, self.frame_repeat)

                if self.SPECTATOR_MODE:
                    sleep(0.065)

                # Update the current state
                s = s_

                if terminate:
                    # Record the total amount of reward in this episode
                    reward = self.game_env.episode_reward()
                    if self.game_env.game.is_player_dead():
                        death += 1
                    self.episode_rewards.append(reward)
                    if self.VERBOSE:
                        print("Total reward: {}.".format(reward), end='\n\n')
        print('death', death)
