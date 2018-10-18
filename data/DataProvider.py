import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from data.celebA import read_data_sets
import tensorflow as tf


class DataProvider:
    @staticmethod
    def get_mnist():
        print('dataset output', input_data.read_data_sets("MNIST_data/", one_hot=True))
        return input_data.read_data_sets("MNIST_data/", one_hot=True), 28

    @staticmethod
    def _parse_(serialized_example):
        feature = {'image_raw': tf.FixedLenFeature([], tf.string)}
        example = tf.parse_single_example(serialized_example, feature)
        image = tf.decode_raw(example['image_raw'],
                              tf.uint8)  # remember to parse in int64. float will raise error
        image = tf.cast(image, tf.float32)
        image = tf.div(image, 255.)
        return image

    @staticmethod
    def get_celeb_a(batch_size):
        # todo Implement this function so it returns the same format as the get_mnist
        # currently we don't need test and validation I guess

        tfrecord_dataset = tf.data.TFRecordDataset("./data/images128.tfrecords")
        # be aware with the cache option; it will cache all data in memory
        tfrecord_dataset = tfrecord_dataset.map(lambda x: DataProvider._parse_(x)).shuffle(True).batch(
            batch_size)

        return tfrecord_dataset, 128


if __name__ == '__main__':
    # DataProvider.get_mnist()
    DataProvider.get_celeb_a(64)
