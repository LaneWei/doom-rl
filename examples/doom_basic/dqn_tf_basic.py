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
from doom_rl.models.tfmodels import SimpleTfModel
from doom_rl.policy import EpsilonGreedyPolicy
from doom_rl.utils import process_gray8_image, process_batch

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
image_gray_scale_level = 32

load_weights = False
log_weights = True

# Log the weights of agent's network after log_weights_epochs epochs
log_weights_epochs = 10

# All weights files are saved to "weight" folder
if not os.path.isdir("weights"):
    os.mkdir("weights")
weights_load_path = join("weights", "dqn_doom_basic.ckpt")
weights_save_path = join("weights", "dqn_doom_basic.ckpt")
config_path = join("..", "configuration", "doom_config", "basic.cfg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    env = DoomGrayEnv(window_length=continuous_frames,
                      process_image=lambda x: process_gray8_image(x, image_shape, image_crop, image_gray_scale_level),
                      configuration_path=config_path)
    nb_buttons = env.available_buttons_size
    action_space = [list(a) for a in it.product([0, 1], repeat=nb_buttons) if a[0] != 1 or a[1] != 1]
    nb_actions = len(action_space)
    model = SimpleTfModel(state_shape=input_shape,
                          nb_actions=nb_actions,
                          process_state_batch=process_batch)
    model.compile(learning_rate)
    agent = DQNAgent(model=model,
                     memory=ListMemory(memory_limit),
                     actions=action_space)

    print("The agent's action space has total {} actions:".format(nb_actions))
    print(action_space, end="\n\n")
    if load_weights:
        agent.model.load_weights(weights_load_path)
        print("Loading model weights from {}.".format(weights_load_path))

    if args.mode == 'train':
        env.set_window_visible(train_visualize)
        training_policy = EpsilonGreedyPolicy(max_eps, min_eps, total_decay_steps=train_epochs * steps_per_epoch)

        total_steps = 0
        time_start_training = time()
        for epoch in range(train_epochs):
            print("\nEpoch {}".format(epoch + 1))
            print("-" * 8)

            s = env.reset()
            epoch_metrics = {"played_episodes": 0,
                             "rewards": [],
                             "epsilons": [],
                             "losses": [],
                             "start_time": time()}
            for learning_step in trange(steps_per_epoch, leave=False):
                total_steps += 1

                # Save the value of current epsilon
                epoch_metrics["epsilons"].append(training_policy.epsilon)

                # Update the policy.
                if total_steps % 100 == 0:
                    training_policy.update(total_steps)

                a = agent.get_action(s, policy=training_policy)
                s_, r, terminate, _ = env.step(action_space[a], frame_repeat=frame_repeat,
                                               reward_discount=True)

                # Save the experience
                agent.save_experience(s, a, r, s_, terminate)
                s = s_

                if terminate:
                    # Save total rewards gained in this episode
                    epoch_metrics["rewards"].append(env.episode_reward())
                    epoch_metrics["played_episodes"] += 1

                    # Reset the environment
                    s = env.reset()

                # Perform learning at the end of each step if it is not warming up
                if total_steps > warm_up_steps:
                    loss = agent.learn_from_memory(batch_size)
                    epoch_metrics["losses"].append(loss)

            # Statistics
            losses_mean = np.mean(epoch_metrics["losses"]) if len(epoch_metrics["losses"]) != 0 else 0
            print("{} training episodes played.".format(epoch_metrics["played_episodes"]))
            print("Agent's memory size: {}".format(agent.memory.size))
            print("mean loss: [{:.3f}]".format(losses_mean), end=' ')
            print("mean epsilon: [{:.3f}]".format(np.mean(epoch_metrics["epsilons"])), end=' ')
            print("mean reward: [{:.2f}±{:.2f}]".format(np.mean(epoch_metrics["rewards"]),
                                                        np.std(epoch_metrics["rewards"])), end=' ')
            print("min: [{:.1f}] max:[{:.1f}]".format(np.min(epoch_metrics["rewards"]),
                                                      np.max(epoch_metrics["rewards"])))
            print("Episode training time: {:.2f} minutes, total training time: {:.2f} minutes.".format(
                 (time() - epoch_metrics["start_time"]) / 60.0, (time() - time_start_training) / 60.0))

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
    env.set_window_visible(test_visualize)
    rewards = []
    for episode in range(test_epochs):
        print('Episode {}:'.format(episode+1))

        s = env.reset()
        terminate = False
        while not terminate:
            a = agent.get_action(s)
            for _ in range(frame_repeat) and not terminate:
                s_, r, terminate, _ = env.step(action_space[a])
                sleep(0.1)
            print('q_values', agent.model.get_q_values(s), end=' ')
            print('action: ', a)

        sleep(1.0)
        reward = env.episode_reward()
        rewards.append(reward)
        print("Total reward: {}.".format(reward), end='\n\n')

    print()
    print("Testing finished, total {} episodes displayed.".format(test_epochs))
    print("mean reward: {:.2f}±{:.2f} min: {:.1f} max:{:.1f}".format(
        np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
