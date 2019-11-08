import argparse
import numpy as np
import tensorflow as tf
from time import time
from tqdm import trange
from os.path import join

from agent import EpsilonGreedyPolicy, A3CTrainAgent, A3CTestAgent
from config import Config
from env import DoomEnv
from network import A3CNetwork
from utils import ThreadDelay, ScalarLogger

if __name__ == '__main__':
    # Argument parser
    config_path = join("..", "doom_configuration", "doom_config", "health_gathering_hard.cfg")
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['train', 'test'], default='test')
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--test_epochs", default=1, type=int)
    parser.add_argument("--test_visible", action="store_true", default=False)
    parser.add_argument("--test_verbose", action="store_true", default=False)
    parser.add_argument("--test_spectator", action="store_true", default=False)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config_path", default=config_path)
    parser.add_argument("--log_path", default="/tmp/log")
    parser.add_argument("--weights_save_path")
    parser.add_argument("--weights_load_dir")
    parser.add_argument("--save_weights_per_epochs", default=5000, type=int)
    args = parser.parse_args()

    epochs = args.epochs
    test_epochs = args.test_epochs
    test_visible = args.test_visible
    test_verbose = args.test_verbose
    test_spectator = args.test_spectator
    training_steps = args.steps
    config_path = args.config_path
    save_path = args.weights_save_path
    load_dir = args.weights_load_dir
    log_path = args.log_path
    save_weights_per_epochs = args.save_weights_per_epochs

    # global network
    graph = tf.Graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config, graph=graph)
    network = A3CNetwork(session=sess, graph=graph)
    if load_dir:
        network.retore(load_dir)
    else:
        network.init_variables()
    network.finalize_graph()

    if args.mode == "train":
        threads = Config.WORKER_THREADS
        file_writer = tf.summary.FileWriter(log_path)
        file_writer.add_graph(graph)
        network.reset_lr()
        envs = [DoomEnv(configuration_path=config_path) for _ in range(threads)]
        decay_steps = 0.2 * epochs * training_steps
        if load_dir:
            train_policies = [
                EpsilonGreedyPolicy(0.01, 0.01, decay_steps) for i in range(threads)
            ]
        else:
            train_policies = [
                EpsilonGreedyPolicy(0.95, 0.01, decay_steps) for i in range(threads)
            ]
        agents = [A3CTrainAgent(network,
                                game_env=envs[i],
                                policy=train_policies[i]) for i in range(threads)]
        # start worker thread
        for agent in agents:
            agent.start()

        metrics = ['learning_rate', 'loss',  'mean_reward', 'max_reward']
        logger = ScalarLogger(metrics, file_writer)
        main_thread_delay = ThreadDelay(1e-3)
        start_time = time()

        # start training
        for epoch in range(epochs):
            print("\nEpoch {} / {}".format(epoch + 1, epochs))
            print("-" * 8)
            losses = []
            epoch_start_time = time()

            # train loop
            for _ in trange(training_steps):
                loss = network.optimize()
                while loss is None:
                    main_thread_delay.delay_on_fail(loss is not None)
                    loss = network.optimize()
                losses.append(loss)

            mean_loss, std_loss = float(np.mean(losses)), float(np.std(losses))
            cur_lr = network.learning_rate
            total = time() - start_time
            h = total // 3600
            epoch_rewards = []

            print("learning rate: [%.3e]" % cur_lr)
            print("mean loss: [%.3f±%.3f]" % (mean_loss, std_loss))
            print('epoch finished in %4.2fm' % ((time() - epoch_start_time) / 60.), end='  ')
            print('total %4.1fh %4.2fm' % (h, (total - h * 3600) / 60.))
            for i in range(len(agents)):
                rewards = agents[i].clear_rewards()
                epoch_rewards.extend(rewards)
                print('Agent %d: %d episodes played, epsilon %.3f' % (i + 1, len(rewards),
                                                                      train_policies[i].get_epsilon()), end=' ')
                print("mean reward: [%.1f±%.1f]" % (float(np.mean(rewards)), float(np.std(rewards))), end=' ')
                print("min: [%.1f] max:[%.1f]" % (float(np.min(rewards)), float(np.max(rewards))))

            mean_reward = float(np.mean(epoch_rewards))
            max_reward = float(np.max(epoch_rewards))

            log_data = [cur_lr, mean_loss, mean_reward, max_reward]
            log_dict = {metrics[i]: log_data[i] for i in range(len(metrics))}
            logger.log_all(log_dict, epoch+1)

            network.lr_decay()
            if (epoch + 1) % save_weights_per_epochs == 0 and save_path:
                network.save_variables(save_path)

        # finish training
        for agent in agents:
            agent.stop()
        for agent in agents:
            agent.join()
        for env in envs:
            env.close()

        print("=" * 15)
        print("Training finished.")
        if save_path:
            network.save_variables(save_path)

    # run test
    if test_epochs > 0:
        test_env = DoomEnv(configuration_path=config_path)
        test_env.set_window_visible(test_visible)
        test_agent = A3CTestAgent(network, test_env, run_episodes=test_epochs,
                                  verbose=test_verbose, spectator_mode=test_spectator)

        print('\nStart testing\n')
        test_agent.run()
        rewards = test_agent.clear_rewards()
        print("{} episodes tested.".format(len(rewards)))
        print("mean reward: [{:.2f}±{:.2f}]".format(np.mean(rewards), np.std(rewards)),
              "min: [{:.1f}] max: [{:.1f}]".format(np.min(rewards), np.max(rewards)))
