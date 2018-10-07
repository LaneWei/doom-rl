from collections import deque
import itertools as it
import numpy as np
from os.path import abspath
import vizdoom as vzd


class DoomGrayEnv:
    """
    The base environment class (screen format has to be GRAY8). It encapsulates vizdoom with some
    of its operations so that a user does not need to interact with vizdoom directly. However,
    a user should interact with vizdoom for higher level control over the doom environment.

    This class contains [gym flavored](https://github.com/openai/gym/blob/master/gym/core.py) APIs
    through which users can interact with the doom environment. In addition, it provides some APIs
    through which users can change the configuration of the doom environment.

    Args:
        window_length: The number of images that will be stacked together as the state of the environment.
        process_image: A function that takes an numpy.ndarray image as input then processes the input
        image and returns an numpy.ndarray.
        configuration_path: Path of the configuration file.

    Properties:
        game: Doom game environment.
        action_space: A list which contains all available actions.
        available_buttons_size: The number of available buttons.
        resolution: A tuple which indicates the shape of screen_buffer.
        screen_buffer: An numpy.ndarray representing the image of the current screen. If the game is not
        running or the current episode is finished, np.zeros(resolution) will be returned.
        processed_screen_image: Processed screen buffer.
    """

    def __init__(self, window_length, process_image, configuration_path=None):
        self.window_length = window_length
        self.process_image = process_image

        self._game = vzd.DoomGame()
        if configuration_path is not None:
            self.load_config(configuration_path)
        self._game.set_screen_format(vzd.ScreenFormat.GRAY8)

        self._action_space = [list(a) for a in it.product([0, 1], repeat=self.available_buttons_size)]
        self._next_state_buffer = None

    def load_config(self, path, verbose=True):
        """
        Load configuration from a configuration file.
        In case of multiple invocations, older configurations will be overwritten by the recent ones.
        Some changes take effect only after the environment restarts. For more information,
        see [vizdoom.load_config](https://github.com/mwydmuch/ViZDoom/blob/master/doc/DoomGame.md#-loadconfig).

        Args:
            path: The path of the configuration file.
            verbose: If True, it will print a status message to console.

        Returns:
            True, if the configuration file is correctly read and applied.
        """

        success = self.game.load_config(path)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        if verbose:
            print("Loading configuration file from {}...\n\n".format(abspath(path)))
        return success

    def reset(self):
        """
        Resets the state of the environment and returns an initial state.
        If the game has not been started, it will start the game. Otherwise, it will start a new episode.
        This method should be called before calling `step()` for the first time.

        Returns:
            The initial state, an numpy.ndarray, whose shape is (h, w, self.window_length).
        """

        if not self.game.is_running():
            self.game.init()
        else:
            self.game.new_episode()

        initial_image = self.processed_screen_image
        self._next_state_buffer = deque([initial_image for _ in range(self.window_length)], self.window_length)
        return np.asarray(self._next_state_buffer, dtype=np.uint8).transpose([1, 2, 0])

    def step(self, action, frame_repeat=1, reward_discount=False):
        """
        Perform one step in the environment. When the end of episode is reached, further calling of this
        method will return undefined result. You need to call `reset()` to reset the environment.

        Args:
            action: Agent's action, which should be contained in the env's action space.
            frame_repeat: The number of frames the agent will skip by taking the same action.
            reward_discount: If True, the reward in this one step will be the original reward
            divided by frame_repeat.

        Returns:
            next_state: The next state, an numpy.ndarray, whose shape is (h, w, self, window_length)
            reward: The amount of reward returned after taking action for frame_repeat frames.
            terminate: A boolean indicates whether the episode has ended.
            info: Diagnostic information. (Nothing will be returned by this `step` method currently. Some game
            variables will be returned in the future)
        """

        assert action in self.action_space
        reward = self.game.make_action(action, frame_repeat)
        if reward_discount:
            reward /= frame_repeat
        terminate = self.game.is_episode_finished()
        next_image = self.processed_screen_image
        self._next_state_buffer.appendleft(next_image)

        return np.asarray(self._next_state_buffer, dtype=np.uint8).transpose([1, 2, 0]), reward, terminate, None

    def episode_reward(self):
        """
        Returns the total number of rewards in one episode.
        """

        return self.game.get_total_reward()

    def close(self):
        """
        Close the environment and performs some cleanups.
        """

        self.game.close()

    def set_window_visible(self, visible):
        """
        See `vizdoom.set_window_visible` for more information.
        """

        self.game.set_window_visible(visible)

    @property
    def game(self):
        return self._game

    @property
    def available_buttons_size(self):
        return self.game.get_available_buttons_size()

    @property
    def action_space(self):
        return list(self._action_space)

    @property
    def resolution(self):
        return self.game.get_screen_height(), self.game.get_screen_width()

    @property
    def screen_buffer(self):
        game_state = self.game.get_state()
        return np.zeros(self.resolution) if game_state is None else game_state.screen_buffer

    @property
    def processed_screen_image(self):
        # returns processed screen buffer
        return self.process_image(self.screen_buffer)
