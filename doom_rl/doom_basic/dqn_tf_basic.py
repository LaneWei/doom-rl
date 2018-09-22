#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import deque
import itertools as it

import numpy as np
from PIL import Image
import random

from tqdm import trange
from time import time, sleep
import vizdoom as vzd

import sys
import os
from os.path import join, dirname
root = dirname(dirname(os.getcwd()))
sys.path.append(root)
from doom_rl.utils.core.memory import ListMemory
from doom_rl.utils.agents import DQNAgent
from doom_rl.utils.models import SimpleTfModel
from doom_rl.utils.policy import EpsilonGreedyPolicy

memory_limit = 40000
learning_rate = 2.5e-4
discount_factor = .99
max_eps = 1.0
min_eps = 0.1

# Total training epochs
train_epochs = 40
steps_per_epoch = 4000
# During warm up, the agent will not perform learning steps
warm_up_steps = 4000
train_visualize = False
test_epochs = 10
test_visualize = True

batch_size = 32

# The number of frames that the agent will skip before taking another action
frame_repeat = 12
# To make sure the agent can acquire more information, the input of the network contains
# most recently continuous frames
continuous_frames = 3

# The height and width of every input image
image_shape = (42, 42)
image_crop = (0, 0, 320, 200)
# The input shape of the network should be (batch_size, height, width, frames)
input_shape = image_shape + (continuous_frames,)
image_rgb_level = 32

load_weights = True
log_weights = True

# Log the weights of agent's network after log_weights_epochs epochs
log_weights_epochs = 10

# All weights files are saved to "weight" folder
if not os.path.isdir("weights"):
    os.mkdir("weights")
weights_load_path = join("weights", "dqn_doom_basic.ckpt")
weights_save_path = join("weights", "dqn_doom_basic.ckpt")
config_path = join(root, "configuration", "doom_config", "basic.cfg")


def image_preprocess(image, shape, crop_box=None, rgb_level=256):
    # format has to be GRAY8
    img = Image.fromarray(image)
    if crop_box is not None:
        # resolution has to be RES320*240
        img = img.crop(crop_box)
    img = img.resize(shape)
    img = np.array(img)
    compress = 256 // rgb_level
    img = img // compress * compress
    return img


def batch_process(batch):
    batch = np.array(batch, dtype=np.float32)
    batch = np.transpose(batch, [0, 2, 3, 1])
    return batch / 255.


def initialize_doom_game(config=None):
    print('\n\nInitializing doom...')
    _game = vzd.DoomGame()
    if config:
        if _game.load_config(config):
            print('Loading configuration file from {}.'.format(config))
        else:
            print('Configuration file path {} does not exist.'.format(config))
    _game.set_window_visible(False)
    _game.set_mode(vzd.Mode.PLAYER)
    # _game.set_screen_format(buffer_format)
    _game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    print('Doom initialized.\n')
    return _game


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    game = initialize_doom_game(config_path)
    nb_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=nb_buttons) if a[0] != 1 or a[1] != 1]
    nb_actions = len(actions)
    print("Total {} actions:".format(nb_actions))
    print(actions)

    agent = DQNAgent(lambda: SimpleTfModel(input_shape, nb_actions, learning_rate, batch_process),
                     ListMemory(memory_limit))
    if load_weights:
        agent.model.load_weights(weights_load_path)

    if args.mode == 'train':
        game.set_window_visible(train_visualize)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.init()
        training_policy = EpsilonGreedyPolicy(max_eps, min_eps, decay_steps=train_epochs*steps_per_epoch)

        total_steps = 0
        time_start_training = time()
        for epoch in range(train_epochs):
            print("\nEpoch {}".format(epoch + 1))
            print("-" * 8)

            train_episodes_finished = 0
            epoch_rewards = []
            epoch_epsilons = []
            epoch_losses = []
            game.new_episode()
            s = image_preprocess(game.get_state().screen_buffer, image_shape, image_crop, rgb_level=image_rgb_level)

            # The current states will be saved in a buffer
            current_state_buffer = deque([s for _ in range(continuous_frames)], maxlen=continuous_frames)
            next_state_buffer = deque([s for _ in range(continuous_frames)], maxlen=continuous_frames)
            time_start_epoch = time()
            for learning_step in trange(steps_per_epoch, leave=False):
                total_steps += 1

                # Save the value of current epsilon
                epoch_epsilons.append(training_policy.epsilon)

                # Update the policy.
                if total_steps % 100 == 0:
                    training_policy.update(total_steps)

                # Get the experience of this training step
                current_state_buffer.appendleft(s)
                action = agent.get_action([list(current_state_buffer)])
                if training_policy.choose_by_random():
                    action = random.choice(range(nb_actions))
                reward = game.make_action(actions[action], frame_repeat)
                terminate = game.is_episode_finished()
                if not terminate:
                    next_state = image_preprocess(game.get_state().screen_buffer, image_shape,
                                                  image_crop, rgb_level=image_rgb_level)
                else:
                    next_state = np.zeros(image_shape)
                next_state_buffer.appendleft(next_state)

                # shrink reward
                reward = reward / frame_repeat

                # Save the experience
                agent.save_experience(list(current_state_buffer), action, reward,
                                      list(next_state_buffer), terminate)
                s = next_state

                if terminate:
                    # Save total rewards gained in this episode
                    epoch_rewards.append(game.get_total_reward())

                    # Reset the environment
                    game.new_episode()
                    train_episodes_finished += 1
                    s = image_preprocess(game.get_state().screen_buffer, image_shape,
                                         image_crop, rgb_level=image_rgb_level)
                    current_state_buffer = deque([s for _ in range(continuous_frames)], maxlen=continuous_frames)
                    next_state_buffer = deque([s for _ in range(continuous_frames)], maxlen=continuous_frames)

                # Perform learning at the end of each step if it is not warming up
                if total_steps > warm_up_steps:
                    loss = agent.learn_from_memory(batch_size)
                    epoch_losses.append(loss)

            # Statistics
            losses_mean = np.mean(epoch_losses) if len(epoch_losses) != 0 else 0
            print("{} training episodes played.".format(train_episodes_finished))
            print("mean loss: [{:.3f}]".format(losses_mean), end=' ')
            print("mean epsilon: [{:.3f}]".format(np.mean(epoch_epsilons)), end=' ')
            print("mean reward: [{:.2f}±{:.2f}]".format(np.mean(epoch_rewards), np.std(epoch_rewards)), end=' ')
            print("min: [{:.1f}] max:[{:.1f}]".format(np.min(epoch_rewards), np.max(epoch_rewards)))
            print("Episode training time: {:.2f} minutes, total training time: {:.2f} minutes.".format(
                 (time() - time_start_epoch) / 60.0, (time() - time_start_training) / 60.0))

            if log_weights and (epoch + 1) % log_weights_epochs == 0:
                print("Saving the network weights to: ", weights_save_path)
                agent.model.save_weights(weights_save_path)

        # Training finished
        game.close()
        print("=" * 15)
        print("Training finished.")
        print("Saving the network weights to: ", weights_save_path)
        agent.model.save_weights(weights_save_path)

    # Start testing
    # The agent now follows greedy policy.
    game.set_window_visible(test_visualize)
    game.init()

    print('\nStart testing\n')
    rewards = []
    for episode in range(test_epochs):
        game.new_episode()
        s = image_preprocess(game.get_state().screen_buffer, image_shape,
                             crop_box=image_crop, rgb_level=image_rgb_level)
        current_state_buffer = deque([s for _ in range(continuous_frames)], maxlen=continuous_frames)
        print('Episode {}:'.format(episode+1))
        while not game.is_episode_finished():
            s = image_preprocess(game.get_state().screen_buffer, image_shape, image_crop, rgb_level=image_rgb_level)
            current_state_buffer.appendleft(s)
            action = agent.get_action([list(current_state_buffer)])
            print('q_values', agent.model.get_q_values([list(current_state_buffer)]), end=' ')
            print('action ', action+1, ' ', actions[action])

            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()
            sleep(0.1)

        sleep(1.0)
        reward = game.get_total_reward()
        rewards.append(reward)
        print("Total reward: {}.".format(reward), end='\n\n')

    print()
    print("Testing finished, total {} episodes displayed.".format(test_epochs))
    print("mean reward: {:.2f}±{:.2f} min: {:.1f} max:{:.1f}".format(
        np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
