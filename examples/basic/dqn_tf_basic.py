#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools as it
import numpy as np
from time import time
from tqdm import trange
import tensorflow as tf
from tensorflow.train import exponential_decay

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
from doom_rl.utils import process_gray8_image, test_model
from doom_rl.utils import ScalarLogger


tf.InteractiveSession()
memory_capacity = 20000
train_epochs = 20
validate_epochs = 20
test_epochs = 10
train_epoch_steps = 4000
train_visualize = False
test_visualize = True
learning_rate = exponential_decay(learning_rate=5e-4,
                                  global_step=tf.train.get_or_create_global_step(),
                                  decay_steps=train_epoch_steps,
                                  decay_rate=0.95)
discount_factor = 0.99

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
log_weights_epochs = 10

# All weights files are saved to "weight" folder
if not os.path.isdir("weights"):
    os.mkdir("weights")
weights_load_path = join("weights", "dqn_doom_basic.ckpt")
weights_save_path = join("weights", "dqn_doom_basic.ckpt")
config_path = join("..", "doom_configuration", "doom_config", "basic.cfg")
summary_log_file = 'log'


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

    print("\nThe agent's action space has {} actions:".format(nb_actions))
    print(action_space, end="\n\n")
    if load_weights:
        agent.model.load_weights(weights_load_path)
        print("Loading model weights from {}.".format(weights_load_path))
        tf.get_default_session().run(tf.assign(tf.train.get_or_create_global_step(), 0, name="InitGlobal"))

    # Start training
    if args.mode == 'train':
        if not os.path.isdir('log'):
            os.mkdir('log')
        tf.summary.FileWriter(summary_log_file, model.session.graph)

        env.set_window_visible(train_visualize)
        metrics = ['learning_rate', 'loss', 'played_episodes', 'mean_epsilon',
                   'mean_train_reward', 'mean_validation_reward']
        logger = ScalarLogger(metrics, summary_log_file)
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
            for _ in trange(train_epoch_steps, leave=False):
                # Update the total training steps in this training epoch
                train_info["steps"] += 1

                # Record the value of the current epsilon
                epsilon = train_info["policy"].epsilon
                epoch_metrics["epsilons"].append(epsilon)

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
                    reward = env.episode_reward()
                    epoch_metrics["rewards"].append(reward)

                    # Update the number of episodes played in this training epoch
                    epoch_metrics["played_episodes"] += 1

                    # Reset the environment
                    s = env.reset()

                # Perform learning step if it is not warming up
                if train_info["steps"] > warm_up_steps:
                    loss = agent.learn_from_memory(batch_size)
                    logger.log('loss', loss, train_info['steps'])
                    epoch_metrics["losses"].append(loss)

            # Statistics
            statistics = {
                'learning_rate': tf.get_default_session().run(learning_rate),
                'mean_loss': np.mean(epoch_metrics['losses']) if len(epoch_metrics["losses"]) != 0 else None,
                'std_loss': np.std(epoch_metrics['losses']) if len(epoch_metrics["losses"]) != 0 else None,
                'mean_epsilon': np.mean(epoch_metrics["epsilons"]),
                'mean_reward': np.mean(epoch_metrics["rewards"]),
                'std_reward': np.std(epoch_metrics["rewards"]),
                'min_reward': np.min(epoch_metrics['rewards']),
                'max_reward': np.max(epoch_metrics['rewards'])
            }
            logger.log('learning_rate', statistics['learning_rate'], epoch)
            logger.log('played_episodes', epoch_metrics["played_episodes"], epoch)
            logger.log('mean_epsilon', statistics['mean_epsilon'], epoch)
            logger.log('mean_train_reward', statistics['mean_reward'], epoch)
            print("{} training episodes played.".format(epoch_metrics["played_episodes"]))
            print("Agent's memory size: {}".format(agent.memory.size))
            print("Learning rate: [{:.3e}]".format(statistics['learning_rate']))
            if len(epoch_metrics["losses"]) != 0:
                print("mean loss: [{:.3f}±{:.3f}]".format(statistics['mean_loss'], statistics['std_loss']), end=' ')
            print("mean epsilon: [{:.3f}]".format(statistics['mean_epsilon']))
            print("mean reward: [{:.2f}±{:.2f}]".format(statistics['mean_reward'], statistics['std_reward']), end=' ')
            print("min: [{:.1f}] max:[{:.1f}]".format(statistics['min_reward'], statistics['max_reward']))

            # Log the weights of the model
            if log_weights and (epoch + 1) % log_weights_epochs == 0:
                print("Saving the network weights to: ", weights_save_path)
                agent.model.save_weights(weights_save_path)

            print("Testing...")
            validation_mean_reward = test_model(env, agent, frame_repeat,
                                                test_epochs=validate_epochs,
                                                test_visualize=False,
                                                verbose=False,
                                                spectator_mode=False)
            logger.log('mean_validation_reward', validation_mean_reward, epoch)
            print("Episode training time: {:.2f} minutes, total training time: {:.2f} minutes.".format(
                (time() - epoch_metrics["start_time"]) / 60.0, (time() - train_info["start_time"]) / 60.0))

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
