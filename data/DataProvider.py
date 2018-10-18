import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from data.celebA import read_data_sets
import tensorflow as tf

class DataProvider:
    @staticmethod
    def get_mnist():
        print('dataset output',input_data.read_data_sets("MNIST_data/", one_hot=True))
        return input_data.read_data_sets("MNIST_data/", one_hot=True), 28

    @staticmethod
    def _parse_(serialized_example):
        feature = {'image_raw': tf.FixedLenFeature([], tf.string)}
        example = tf.parse_single_example(serialized_example, feature)
        image = tf.decode_raw(example['image_raw'], tf.int8)  # remember to parse in int64. float will raise error
        return dict({'image': image})

    @staticmethod
    def get_celeb_a(batch_size):
        # todo Implement this function so it returns the same format as the get_mnist
        # currently we don't need test and validation I guess

        tfrecord_dataset = tf.data.TFRecordDataset("./data/images128.tfrecords")

        tfrecord_dataset = tfrecord_dataset.map(lambda x: DataProvider._parse_(x)).shuffle(True).batch(batch_size)

        tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()

        return tfrecord_iterator, 128


if __name__ == '__main__':
    #DataProvider.get_mnist()
    DataProvider.get_celeb_a(64)
