import pytest

import numpy as np
from doom_rl.utils import process_gray8_image, process_batch

np.random.seed(1)


def get_gray8_img(shape):
    return np.random.randint(256, size=shape, dtype=np.uint8)


def test_process_gray8_image():
    img_shape = (320, 480)
    output_shape = (64, 64)
    gray_scale_level = 16

    img = get_gray8_img(img_shape)
    assert np.unique(img).size == 256

    processed_img = process_gray8_image(img, output_shape, gray_scale_level=gray_scale_level)
    assert processed_img.shape == output_shape
    assert np.unique(processed_img).size == gray_scale_level


def test_process_batch():
    img_shape = (320, 480)
    batch = np.asarray([get_gray8_img(img_shape) for _ in range(64)])

    processed_batch = process_batch(batch)
    for img in processed_batch:
        assert img.dtype == np.float32
        assert np.all(img >= 0) and np.all(img <= 1)


if __name__ == "__main":
    pytest.main(__file__)
