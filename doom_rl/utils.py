import numpy as np
from PIL import Image


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
