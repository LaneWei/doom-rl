import numpy as np
from PIL import Image
from time import sleep, time
from tqdm import trange
import vizdoom as vzd

from doom_rl.policy import GreedyPolicy


def process_gray8_image(image, output_shape, crop_box=None, gray_scale_level=256):
    """
    Process GRAY8 format image  .

    Args:
        image: An numpy.ndarray representing the input image.
        output_shape: The shape of the output image.
        crop_box: A crop rectangle defined by a 4-tuple which specifies the left, upper, right,
        and lower pixel coordinate. The input image will first be cropped before resizing.
        gray_scale_level: A integer, ranging from 1 to 256, that defines the gray scale level of the output image.

    Returns:
        The processed GRAY8 image.
    """
    if gray_scale_level < 1 or gray_scale_level > 256:
        raise ValueError('Gray scale level should be an integer ranging from 1 to 256')

    img = Image.fromarray(image)
    if crop_box is not None:
        img = img.crop(crop_box)
    img = img.resize(output_shape)
    img = np.array(img, dtype=np.uint8)

    compress_level = 256 // gray_scale_level
    img = img // compress_level * compress_level
    return img


def process_batch(batch):
    """
    Process a batch of images.

    Args:
        batch: The batch of images.

    Returns:
        The processed batch of images.
    """

    batch = np.array(batch, dtype=np.float32)
    return batch / 255.


def train_model(env,
                agent,
                frame_repeat,
                batch_size,
                train_epochs,
                train_epoch_steps,
                train_policy,
                validate_epochs,
                weights_save_path,
                warm_up_steps=0,
                train_visualize=False,
                train_verbose=True,
                validate_policy=GreedyPolicy()):
    env.set_window_visible(train_visualize)
    train_info = {"policy": train_policy,
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
        for _ in trange(train_epoch_steps, leave=False) if train_verbose else range(train_epoch_steps):
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
        if (epoch + 1) % 5 == 0:
            print("Saving the network weights to: ", weights_save_path)
            agent.model.save_weights(weights_save_path)

        test_model(env, agent, frame_repeat,
                   test_epochs=validate_epochs,
                   test_policy=validate_policy,
                   test_visualize=False,
                   verbose=False,
                   spectator_mode=False)


def test_model(env,
               agent,
               frame_repeat,
               test_epochs,
               test_policy=GreedyPolicy(),
               test_visualize=True,
               verbose=True,
               spectator_mode=False):
    """
    Perform testing.

    Args:
        env: The environment.
        agent: The agent.
        frame_repeat: The number of frames skipped after the agent takes an action.
        test_epochs: The number of epochs to be tested.
        test_policy: The policy that the agent will use during this test.
        test_visualize: Visualize the test.
        verbose: Print verbose information (q_values, action, reward).
        spectator_mode: If true,

    Returns:
        Mean rewards gained during test_epochs of game epochs.
    """

    if spectator_mode:
        env.game.set_mode(vzd.Mode.SPECTATOR)
    env.set_window_visible(test_visualize)

    rewards = []
    if verbose:
        for episode in range(test_epochs):
            print('Episode {}:'.format(episode+1))

            s = env.reset()
            terminate = False
            while not terminate:
                a = agent.get_action(s, policy=test_policy)
                a_id = agent.get_action_id(a)
                reward = 0
                for _ in range(frame_repeat):
                    s, r, terminate, _ = env.step(a)
                    reward += r
                    if terminate:
                        break
                    sleep(0.015)
                if not spectator_mode:
                    print('q_values', agent.get_q_values(s), end=' ')
                    print('action: ', a_id + 1, ", ", a, end=' ')
                print('reward: ', reward)

            sleep(1.0)
            reward = env.episode_reward()
            rewards.append(reward)
            print("Total reward: {}.".format(reward), end='\n\n')
    else:
        for _ in trange(test_epochs, leave=False):
            s = env.reset()
            terminate = False
            while not terminate:
                a = agent.get_action(s, policy=test_policy)
                reward = 0
                for _ in range(frame_repeat):
                    s, r, terminate, _ = env.step(a)
                    reward += r
                    if terminate:
                        break
            rewards.append(env.episode_reward())
            if spectator_mode:
                sleep(1.0)

    print()
    print("Testing finished, total {} episodes displayed.".format(test_epochs))
    print("mean reward: {:.2f}±{:.2f} min: {:.1f} max:{:.1f}".format(np.mean(rewards),
                                                                     np.std(rewards),
                                                                     np.min(rewards),
                                                                     np.max(rewards)))
    return np.mean(rewards)
