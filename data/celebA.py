import numpy as np
import cv2
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.python.util.deprecation import deprecated

class DataSet(object):
  """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate(
          (images_rest_part, images_new_part), axis=0), np.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir):

  train_images = np.load('images128.npy')
  train_images = np.swapaxes(train_images, 1, 2)
  train_images = np.swapaxes(train_images, 2, 3)
  img_size = len(train_images[0])
  print(train_images.shape)
  print(img_size)
  print('train_images shape: ', train_images.shape)

  #use one channel as test:
  train_images = train_images[:,:,:,0]
  to_one_ch = train_images[..., np.newaxis]
  train_images = to_one_ch.reshape((len(train_images),img_size,img_size,1))
  #reduce the nr of images:
  #train_images = train_images[:6000,:,:,:]
  #np.save('test3.npy', train_images)

  #for i in range(len(train_images)):
    #train_images[i] = cv2.resize(train_images[i,:,:,:], (28, 28), interpolation=cv2.INTER_CUBIC)

  print("train img celeb: ", train_images.shape)

  #validation_size = 5000
  #validation_images = (5000, 28, 28, 1)
  #validation_labels = (5000, 10)
  #train_labels = (55000, 10)
  #train_images = (55000, 28, 28, 1)
  #test_labels = (10000,10)
  #test_images = (10000, 28, 28, 1)
  validation_size = 5000
  nr_classes = 10

  #test_labels = np.zeros((1000, 10))
  #train_labels = np.zeros((5500,10))
  #test_images = np.zeros((1000, 128, 128, 1))

  test_labels = np.zeros((validation_size,nr_classes))
  train_labels = np.zeros((train_images.shape[0],nr_classes))
  test_images = np.zeros((validation_size,img_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  #train_images = train_images[validation_size:]
  #train_labels = train_labels[validation_size:]

  dtype = dtypes.float32
  reshape = True
  seed = None

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)


def load_celebA(train_dir='celebA-data'):
  return read_data_sets(train_dir)