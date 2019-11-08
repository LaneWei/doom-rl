import cv2
import numpy as np
import tensorflow as tf
from time import sleep
from random import randint


def process_batch(batch):
    batch = np.divide(batch, 255., dtype=np.float32)
    return batch


def process_image(image, output_shape=None, crop_box=None, gray_scale_level=256):
    if crop_box is not None:
        left, upper, right, lower = crop_box
        image = image[upper:lower, left:right]
    if output_shape is not None:
        image = cv2.resize(image, output_shape)

    comp_level = 256 // gray_scale_level
    image = np.floor_divide(image, comp_level, dtype=image.dtype)
    image = np.multiply(image, comp_level, dtype=image.dtype)
    return image


class ThreadDelay:
    def __init__(self, delay_time, max_count=10):
        self.delay_time = delay_time
        self.max_count = max_count
        self.fail_count = 0

    def delay_on_fail(self, success):
        if success:
            self.fail_count = 0
            return
        self.fail_count += 1
        self.fail_count = min(self.fail_count, self.max_count)
        n = randint(0, self.fail_count)
        sleep(n * self.delay_time)


class ScalarLogger:
    def __init__(self, metrics, file_writer):
        self.metrics = metrics
        with tf.Graph().as_default():
            self.sess = tf.Session()
            self.writer = file_writer
            with tf.name_scope('Summaries'):
                self._placeholders = {m: tf.placeholder(tf.float32, name=m) for m in metrics}
                self._summaries = {m: tf.summary.scalar(m, self._placeholders[m]) for m in metrics}
                self.merge_all = tf.summary.merge_all()

    def log(self, name, scalar, step):
        placeholder = self._placeholders[name]
        summary = self._summaries[name]
        self.writer.add_summary(self.sess.run(summary, {placeholder: scalar}), step)

    def log_all(self, names_to_data, step):
        feed_dict = {self._placeholders[name]: data for name, data in names_to_data.items()}
        sum_all = self.sess.run(self.merge_all, feed_dict)
        self.writer.add_summary(sum_all, step)
