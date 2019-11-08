import vizdoom as vzd
import numpy as np
from time import sleep
from os.path import abspath

from config import Config
from utils import process_image


def process_screen(img):
    img_shape = (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
    img_crop_box = Config.IMAGE_CROP_BOX
    img_gray_level = Config.IMAGE_GRAY_LEVEL
    return process_image(img, img_shape, img_crop_box, img_gray_level)


class DoomEnv:
    def __init__(self, configuration_path=None):
        self.game = vzd.DoomGame()
        if configuration_path is not None:
            self.load_config(configuration_path)

        self.action_space = Config.ACTION_SPACE
        self._resolution = None

    def get_action(self, index):
        return self.action_space[index]

    def load_config(self, path):
        # new configuration takes effect after game init
        print("Loading configuration file from %s..." % abspath(path))
        return self.game.load_config(path)

    def close(self):
        self.game.close()

    def reset(self):
        if self.game.is_running():
            self.game.new_episode()
        else:
            self.game.init()

        self._resolution = None
        initial_image = self.get_processed_screen_image()
        return initial_image

    def step(self, action_index, frame_repeat=1, slow_update=False, reward_discount=1):
        action = self.action_space[action_index]
        reward = 0
        # smooth update
        if slow_update:
            for i in range(frame_repeat):
                reward += self.game.make_action(action, 1)
                sleep(0.003)
        else:
            reward = self.game.make_action(action, frame_repeat)
        if reward_discount > 1:
            reward /= reward_discount
        terminate = self.game.is_episode_finished()
        next_image = self.get_processed_screen_image()
        return next_image, reward, terminate

    def episode_reward(self):
        return self.game.get_total_reward()

    def set_window_visible(self, visible):
        self.game.set_window_visible(visible)

    @property
    def screen_resolution(self):
        if self._resolution is None:
            screen_height = self.game.get_screen_height()
            screen_width = self.game.get_screen_width()
            screen_format = self.game.get_screen_format()
            if screen_format == vzd.ScreenFormat.GRAY8 or screen_format == vzd.ScreenFormat.DOOM_256_COLORS8:
                self._resolution = (screen_height, screen_width)
            else:
                self._resolution = (screen_height, screen_width, 3)
        return self._resolution

    def get_screen_buffer(self):
        game_state = self.game.get_state()
        if game_state is None:
            screen = np.zeros(self.screen_resolution, dtype=np.uint8)
        else:
            screen = game_state.screen_buffer
        return screen

    def get_processed_screen_image(self):
        image = self.get_screen_buffer()
        if image is not None:
            image = process_screen(image)
        return image
