import numpy as np
from PIL import Image
from time import sleep
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
        for _ in trange(test_epochs):
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
    print("mean reward: {:.2f}±{:.2f} min: {:.1f} max:{:.1f}".format(
        np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)))
