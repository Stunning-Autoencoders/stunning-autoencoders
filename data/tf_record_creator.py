import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_records(x, path):
    print('writing to {}'.format(path))
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in range(x.shape[0]):
            example = tf.train.Example(features=tf.train.Features(
                feature=
                {
                    'image_raw': _bytes_feature(x[i].tostring())
                }
            )
            )
            writer.write(example.SerializeToString())
            if i % 5000 == 0:
                print('writing {}th image'.format(i))


if __name__ == '__main__':
    data = np.load("images128.npy")
    convert_to_records(data, "images128.tfrecords")
