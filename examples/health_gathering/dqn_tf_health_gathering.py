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
from doom_rl.utils import process_gray8_image, test_model, train_model
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

train_epochs = 80
validate_epochs = 5
test_epochs = 3
train_epoch_steps = 4000
train_visualize = False
test_visualize = True

# Epsilon greedy policy settings.
max_eps = 0.8
min_eps = 0.1
decay_steps = train_epochs * train_epoch_steps

# During warm up, the agent will not perform learning steps
warm_up_steps = 4000

# The number of steps before the target network is updated
update_steps = 10000

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
# log_weights_epochs = 5

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

    print("\n\nThe agent's action space has {} actions:".format(nb_actions))
    print(action_space, end="\n\n")
    if load_weights:
        agent.model.load_weights(weights_load_path)
        print("Loading model weights from {}.".format(weights_load_path))

    # Start training
    if args.mode == 'train':
        train_model(env,
                    agent,
                    frame_repeat,
                    batch_size,
                    train_epochs=train_epochs,
                    train_epoch_steps=train_epoch_steps,
                    train_policy=EpsilonGreedyPolicy(max_eps, min_eps, total_decay_steps=decay_steps),
                    validate_epochs=validate_epochs,
                    weights_save_path=weights_save_path,
                    warm_up_steps=warm_up_steps,
                    train_visualize=train_visualize,
                    train_verbose=True)

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
