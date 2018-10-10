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
os.chdir(dirname(os.path.abspath(__file__)))
root_path = dirname(dirname(os.getcwd()))
sys.path.append(root_path)

from doom_rl.agents.dqn import DQNAgent
from doom_rl.envs.env import DoomGrayEnv
from doom_rl.memory import ListMemory
from doom_rl.models.model import DqnTfModel
from doom_rl.policy import EpsilonGreedyPolicy
from doom_rl.utils import process_gray8_image, test_model
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.layers import conv2d, dense
from tensorflow.nn import relu


class HGModel(DqnTfModel):
    def __init__(self, state_shape, n_actions, discount_factor):
        super(HGModel, self).__init__(state_shape, n_actions, discount_factor)

    def _build_network(self):
        conv1 = conv2d(self.s_input, 16, 8, strides=(4, 4), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer1')
        conv2 = conv2d(conv1, 24, 5, strides=(3, 3), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer2')
        conv3 = conv2d(conv2, 32, 3, strides=(1, 1), activation=relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                       bias_initializer=tf.constant_initializer(0.01), name='ConvLayer3')
        conv_flat = flatten(conv3)
        fc1 = dense(conv_flat, 64, activation=relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.constant_initializer(0.01), name='FullyConnected1')

        output_layer = dense(fc1, self.nb_actions, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=tf.constant_initializer(0.01), name='QValues')
        return output_layer


memory_capacity = 20000
learning_rate = 2.5e-4
gamma = 0.95

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

batch_size = 32

# The number of frames that the agent will skip before taking another action
frame_repeat = 8

# To make sure the agent can acquire more information, the input of the network contains
# most recently continuous frames
continuous_frames = 3

# The height and width of every input image
image_shape = (96, 96)

# A rectangular region of the image to be cropped
image_crop = (0, 80, 640, 410)

# The input shape of the network should be (batch_size, height, width, frames)
input_shape = image_shape + (continuous_frames,)
image_gray_scale_level = 16

load_weights = False
log_weights = True

# Log the weights of agent's network after log_weights_epochs epochs
log_weights_epochs = 5

# All weights files are saved to "weight" folder
if not os.path.isdir("weights"):
    os.mkdir("weights")
weights_load_path = join("weights", "dqn_health_gathering_hard.ckpt")
weights_save_path = join("weights", "dqn_health_gathering_hard.ckpt")
config_path = join("..", "doom_configuration", "doom_config", "health_gathering.cfg")


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
                    discount_factor=gamma)

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

    # Start testing
    print('\nStart testing\n')
    test_model(env, agent, frame_repeat,
               test_epochs=test_epochs,
               test_visualize=test_visualize,
               verbose=True,
               spectator_mode=False)
