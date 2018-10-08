#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools as it
import numpy as np
from tqdm import trange
from time import time, sleep

import sys
import os
from os.path import join, dirname
os.chdir(dirname(__file__))
root_path = dirname(dirname(os.getcwd()))
sys.path.append(root_path)

from doom_rl.agents.dqn import DQNAgent
from doom_rl.envs.env import DoomGrayEnv
from doom_rl.memory import ListMemory
from doom_rl.models.model import DqnTfModel
from doom_rl.policy import EpsilonGreedyPolicy
from doom_rl.utils import process_gray8_image, process_batch
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.layers import conv2d, dense
from tensorflow.nn import relu


class HGModel(DqnTfModel):
    def __init__(self, state_shape, n_actions, process_state_batch):
        super(HGModel, self).__init__(state_shape, n_actions, process_state_batch=process_state_batch)

    def _build_network(self):
        conv1 = conv2d(self.s_input, 12, 8, strides=(4, 4), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer1')
        conv2 = conv2d(conv1, 24, 4, strides=(2, 2), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer2')
        conv3 = conv2d(conv2, 32, 3, strides=(2, 2), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer3')
        conv_flat = flatten(conv3)
        fc1 = dense(conv_flat, 64, activation=relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.constant_initializer(0.01), name='FullyConnected1')

        self.q_values = dense(fc1, self.nb_actions, activation=None,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.constant_initializer(0.01))


memory_capacity = 20000
learning_rate = 2.5e-4
discount_factor = 0.95

# Training epochs
train_epochs = 80
steps_per_epoch = 4000

# Epsilon greedy policy settings.
max_eps = 0.8
min_eps = 0.1
decay_steps = train_epochs * steps_per_epoch

# During warm up, the agent will not perform learning steps
warm_up_steps = 4000
train_visualize = False
test_epochs = 3
test_visualize = True

batch_size = 24

# The number of frames that the agent will skip before taking another action
frame_repeat = 8

# To make sure the agent can acquire more information, the input of the network contains
# most recently continuous frames
continuous_frames = 3

# The height and width of every input image
image_shape = (84, 84)

# A rectangular region of the image to be cropped
image_crop = (0, 60, 640, 410)

# The input shape of the network should be (batch_size, height, width, frames)
input_shape = image_shape + (continuous_frames,)
image_gray_scale_level = 64

load_weights = True
log_weights = True

# Log the weights of agent's network after log_weights_epochs epochs
log_weights_epochs = 5

# All weights files are saved to "weight" folder
if not os.path.isdir("weights"):
    os.mkdir("weights")
weights_load_path = join("weights", "dqn_health_gathering_hard.ckpt")
weights_save_path = join("weights", "dqn_health_gathering_hard.ckpt")
config_path = join("..", "configuration", "doom_config", "health_gathering_hard.cfg")


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    # The doom game environment
    env = DoomGrayEnv(window_length=continuous_frames,
                      process_image=lambda x: process_gray8_image(x, image_shape, image_crop, image_gray_scale_level),
                      configuration_path=config_path)

    # The number of buttons available
    nb_buttons = env.available_buttons_size

    # The action space contains all available actions the agent can perform
    action_space = [list(a) for a in it.product([0, 1], repeat=nb_buttons) if a[0] != 1 or a[1] != 1]
    nb_actions = len(action_space)

    # Agent's model (pure tensorflow)
    model = HGModel(state_shape=input_shape,
                    n_actions=nb_actions,
                    process_state_batch=process_batch)

    # Before using a model, it has to be compiled
    model.compile(learning_rate)

    # The agent (Q-learning)
    agent = DQNAgent(model=model,
                     memory=ListMemory(memory_capacity),
                     actions=action_space)

    # Expected sarsa
    # agent = ESarsaAgent(model=model,
    #                     memory=ListMemory(memory_capacity),
    #                     actions=action_space)

    print("\n\nThe agent's action space has {} actions:".format(nb_actions))
    print(action_space, end="\n\n")
    if load_weights:
        agent.model.load_weights(weights_load_path)
        print("Loading model weights from {}.".format(weights_load_path))

    # Start training
    if args.mode == 'train':
        env.set_window_visible(train_visualize)
        train_info = {"policy": EpsilonGreedyPolicy(max_eps, min_eps, total_decay_steps=decay_steps),
                      "steps": 0,
                      "start_time": time()}
        for epoch in range(train_epochs):
            print("\nEpoch {}".format(epoch + 1))
            print("-" * 8)

            s = env.reset()
            epoch_metrics = {"played_episodes": 0,
                             "rewards": [],
                             "epsilons": [],
                             "losses": [],
                             "start_time": time()}

            # Perform one training epoch
            for learning_step in trange(steps_per_epoch, leave=False):
                # Update the total training steps in this training epoch
                train_info["steps"] += 1

                # Record the value of the current epsilon
                epoch_metrics["epsilons"].append(train_info["policy"].epsilon)

                # Update the policy every 100 training steps
                if train_info["steps"] % 100 == 0:
                    train_info["policy"].update(train_info["steps"])

                # Get the agent's action and its id
                a = agent.get_action(s, policy=train_info["policy"])
                a_id = agent.get_action_id(a)

                # Take one step in the environment
                s_, r, terminate, _ = env.step(a, frame_repeat=frame_repeat, reward_discount=True)

                # Save this experience
                agent.save_experience(s, a_id, r, s_, terminate)

                # Update the current state
                s = s_

                if terminate:
                    # Record the total amount of reward in this episode
                    epoch_metrics["rewards"].append(env.episode_reward())

                    # Update the number of episodes played in this training epoch
                    epoch_metrics["played_episodes"] += 1

                    # Reset the environment
                    s = env.reset()

                # Perform learning step if it is not warming up
                if train_info["steps"] > warm_up_steps:
                    loss = agent.learn_from_memory(batch_size)
                    epoch_metrics["losses"].append(loss)

            # Statistics
            print("{} training episodes played.".format(epoch_metrics["played_episodes"]))
            print("Agent's memory size: {}".format(agent.memory.size))
            if len(epoch_metrics["losses"]) != 0:
                print("mean loss: [{:.3f}±{:.3f}]".format(np.mean(epoch_metrics["losses"]),
                                                          np.std(epoch_metrics["losses"])), end=' ')
            print("mean epsilon: [{:.3f}]".format(np.mean(epoch_metrics["epsilons"])))

            print("mean reward: [{:.2f}±{:.2f}]".format(np.mean(epoch_metrics["rewards"]),
                                                        np.std(epoch_metrics["rewards"])), end=' ')
            print("min: [{:.1f}] max:[{:.1f}]".format(np.min(epoch_metrics["rewards"]),
                                                      np.max(epoch_metrics["rewards"])))
            print("Episode training time: {:.2f} minutes, total training time: {:.2f} minutes.".format(
                 (time() - epoch_metrics["start_time"]) / 60.0, (time() - train_info["start_time"]) / 60.0))

            # Log the weights of the model
            if log_weights and (epoch + 1) % log_weights_epochs == 0:
                print("Saving the network weights to: ", weights_save_path)
                agent.model.save_weights(weights_save_path)

        # Training finished
        env.close()
        print("=" * 15)
        print("Training finished.")
        print("Saving the network weights to: ", weights_save_path)
        agent.model.save_weights(weights_save_path)

    print('\nStart testing\n')
    # Start testing
    # The agent now follows greedy policy.
    # import vizdoom as vzd
    # env.game.set_mode(vzd.Mode.SPECTATOR)
    env.set_window_visible(test_visualize)
    rewards = []
    for episode in range(test_epochs):
        print('Episode {}:'.format(episode+1))

        s = env.reset()
        terminate = False
        while not terminate:
            a = agent.get_action(s)
            a_id = agent.get_action_id(a)
            reward = 0
            for _ in range(frame_repeat):
                s, r, terminate, _ = env.step(a)
                reward += r
                if terminate:
                    break
                sleep(0.015)
            print('q_values', agent.get_q_values(s), end=' ')
            print('action: ', a_id + 1, ", ", a, end=' ')
            print('reward: ', reward)

        sleep(1.0)
        reward = env.episode_reward()
        rewards.append(reward)
        print("Total reward: {}.".format(reward), end='\n\n')

    print()
    print("Testing finished, total {} episodes displayed.".format(test_epochs))
    print("mean reward: {:.2f}±{:.2f} min: {:.1f} max:{:.1f}".format(
        np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
