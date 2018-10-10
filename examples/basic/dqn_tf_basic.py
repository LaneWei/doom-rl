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
# from doom_rl.agents.sarsa import ESarsaAgent
from doom_rl.envs.env import DoomGrayEnv
from doom_rl.memory import ListMemory
from doom_rl.models.tfmodels import SimpleTfModel
from doom_rl.policy import EpsilonGreedyPolicy
from doom_rl.utils import process_gray8_image, test_model, train_model

memory_capacity = 20000
learning_rate = 4.5e-4
discount_factor = 0.99

train_epochs = 40
validate_epochs = 50
test_epochs = 10
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
frame_repeat = 12

# To make sure the agent can acquire more information, the input of the network contains
# most recently continuous frames
continuous_frames = 2

# The height and width of every input image
image_shape = (42, 42)

# A rectangular region of the image to be cropped
image_crop = (0, 0, 320, 200)

# The input shape of the network should be (batch_size, height, width, frames)
input_shape = image_shape + (continuous_frames,)
image_gray_scale_level = 16

load_weights = False
log_weights = True

# Log the weights of agent's network after log_weights_epochs epochs
# log_weights_epochs = 10

# All weights files are saved to "weight" folder
if not os.path.isdir("weights"):
    os.mkdir("weights")
weights_load_path = join("weights", "dqn_doom_basic.ckpt")
weights_save_path = join("weights", "dqn_doom_basic.ckpt")
config_path = join("..", "doom_configuration", "doom_config", "basic.cfg")


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
    model = SimpleTfModel(state_shape=input_shape,
                          nb_actions=nb_actions,
                          discount_factor=discount_factor,
                          update_steps=update_steps)

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
