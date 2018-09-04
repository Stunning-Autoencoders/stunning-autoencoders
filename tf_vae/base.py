#!/usr/bin/env python

"""
@author: Dan Salo, Nov 2016
Purpose: To facilitate network creation, data I/O, and model training in TensorFlow
Classes:
    Layers
    Data
    Model
"""

import tensorflow as tf
import numpy as np
import logging
import tensorflow.contrib.layers as init
import math
import os
import datetime


class Layers:
    """
    A Class to facilitate network creation in TensorFlow.
    Methods: conv2d, deconv2d, cflatten, maxpool, avgpool, res_layer, noisy_and, batch_norm
    """

    def __init__(self, x):
        """
        Initialize model Layers.
        .input = numpy array
        .count = dictionary to keep count of number of certain types of layers for naming purposes
        """
        self.input = x  # initialize input tensor
        self.count = {'conv': 0, 'deconv': 0, 'fc': 0, 'flat': 0, 'mp': 0, 'up': 0, 'ap': 0, 'rn': 0}

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0,
               s_value=1.0, bn=True):
        """
        2D Convolutional Layer.
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        self.count['conv'] += 1
        scope = 'conv_' + str(self.count['conv'])
        with tf.variable_scope(scope):

            # Conv function
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, input_channels, output_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding=padding)

            # Additional functions
            if bn is True:  # batch normalization
                self.input = self.batch_norm(self.input)
            if b_value is not None:  # bias value
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:  # scale value
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = tf.multiply(self.input, s)
            if activation_fn is not None:  # activation function
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def deconv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0,
                 s_value=1.0, bn=True):
        """
        2D Deconvolutional Layer
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        self.count['deconv'] += 1
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):

            # Calculate the dimensions for deconv function
            batch_size = tf.shape(self.input)[0]
            input_height = tf.shape(self.input)[1]
            input_width = tf.shape(self.input)[2]

            if padding == "VALID":
                out_rows = (input_height - 1) * stride + filter_size
                out_cols = (input_width - 1) * stride + filter_size
            else:  # padding == "SAME":
                out_rows = input_height * stride
                out_cols = input_width * stride

            # Deconv function
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, output_channels, input_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            deconv_out_shape = tf.stack([batch_size, out_rows, out_cols, output_channels])
            self.input = tf.nn.conv2d_transpose(self.input, w, deconv_out_shape, [1, stride, stride, 1], padding)

            # Additional functions
            if bn is True:  # batch normalization
                self.input = self.batch_norm(self.input)
            if b_value is not None:  # bias value
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:  # scale value
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = tf.multiply(self.input, s)
            if activation_fn is not None:  # non-linear activation function
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))  # print shape of output

    def flatten(self, keep_prob=1):
        """
        Flattens 4D Tensor (from Conv Layer) into 2D Tensor (to FC Layer)
        :param keep_prob: int. set to 1 for no dropout
        """
        self.count['flat'] += 1
        scope = 'flat_' + str(self.count['flat'])
        with tf.variable_scope(scope):
            # Reshape function
            input_nodes = tf.Dimension(
                self.input.get_shape()[1] * self.input.get_shape()[2] * self.input.get_shape()[3])
            output_shape = tf.stack([-1, input_nodes])
            self.input = tf.reshape(self.input, output_shape)

            # Dropout function
            if keep_prob != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def fc(self, output_nodes, keep_prob=1, activation_fn=tf.nn.relu, b_value=0.0, s_value=None, bn=False):
        """
        Fully Connected Layer
        :param output_nodes: int
        :param keep_prob: int. set to 1 for no dropout
        :param activation_fn: tf.nn function
        :param b_value: float or None
        :param s_value: float or None
        :param bn: bool
        """
        self.count['fc'] += 1
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):

            # Matrix Multiplication Function
            input_nodes = self.input.get_shape()[1]
            output_shape = [input_nodes, output_nodes]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.matmul(self.input, w)

            # Additional Functions
            if bn is True:  # batch normalization
                self.input = self.batch_norm(self.input, 'fc')
            if b_value is not None:  # bias value
                b = self.const_variable(name='bias', shape=[output_nodes], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:  # scale value
                s = self.const_variable(name='scale', shape=[output_nodes], value=s_value)
                self.input = tf.multiply(self.input, s)
            if activation_fn is not None:  # activation function
                self.input = activation_fn(self.input)
            if keep_prob != 1:  # dropout function
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def maxpool(self, k=2, globe=False):
        """
        Takes max value over a k x k area in each input map, or over the entire map (global = True)
        :param k: int
        :param globe:  int, whether to pool over each feature map in its entirety
        """
        self.count['mp'] += 1
        scope = 'maxpool_' + str(self.count['mp'])
        with tf.variable_scope(scope):
            if globe is True:  # Global Pool Parameters
                k1 = self.input.get_shape()[1]
                k2 = self.input.get_shape()[2]
                s1 = 1
                s2 = 1
                padding = 'VALID'
            else:
                k1 = k
                k2 = k
                s1 = k
                s2 = k
                padding = 'SAME'
            # Max Pool Function
            self.input = tf.nn.max_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def avgpool(self, k=2, globe=False):
        """
        Averages the values over a k x k area in each input map, or over the entire map (global = True)
        :param k: int
        :param globe: int, whether to pool over each feature map in its entirety
        """
        self.count['ap'] += 1
        scope = 'avgpool_' + str(self.count['mp'])
        with tf.variable_scope(scope):
            if globe is True:  # Global Pool Parameters
                k1 = self.input.get_shape()[1]
                k2 = self.input.get_shape()[2]
                s1 = 1
                s2 = 1
                padding = 'VALID'
            else:
                k1 = k
                k2 = k
                s1 = k
                s2 = k
                padding = 'SAME'
            # Average Pool Function
            self.input = tf.nn.avg_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def res_layer(self, output_channels, filter_size=3, stride=1, activation_fn=tf.nn.relu):
        """
        Residual Layer: Input -> BN, Act_fn, Conv1, BN, Act_fn, Conv 2 -> Output.  Return: Input + Output
        If stride > 1, decrease dims of Input by passing through a 1 x 1 Conv Layer
        :param output_channels: int
        :param filter_size: int. assumes square filter
        :param stride: int
        :param activation_fn: tf.nn function
        """
        self.count['rn'] += 1
        scope = 'resnet_' + str(self.count['rn'])
        input_channels = self.input.get_shape()[3]
        with tf.variable_scope(scope):

            # Determine Additive Output Based on Stride in First Conv Layer.
            if stride != 1:  # Decrease Input dimension with 1 x 1 Conv Layer with stride > 1
                with tf.variable_scope('conv0'):
                    output_shape = [1, 1, input_channels, output_channels]
                    w = self.weight_variable(name='weights', shape=output_shape)
                    additive_output = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding='SAME')
                    b = self.const_variable(name='bias', shape=[output_channels], value=0.0)
                    additive_output = tf.add(additive_output, b)
            else:
                additive_output = self.input

            # First Conv Layer. Implement stride in this layer if desired.
            with tf.variable_scope('conv1'):
                output_shape = [filter_size, filter_size, input_channels, output_channels]
                w = self.weight_variable(name='weights', shape=output_shape)
                self.input = self.batch_norm(self.input)
                self.input = activation_fn(self.input)
                self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding='SAME')
                b = self.const_variable(name='bias', shape=[output_channels], value=0.0)
                self.input = tf.add(self.input, b)
            # Second Conv Layer
            with tf.variable_scope('conv2'):
                input_channels = self.input.get_shape()[3]
                output_shape = [filter_size, filter_size, input_channels, output_channels]
                w = self.weight_variable(name='weights', shape=output_shape)
                self.input = self.batch_norm(self.input)
                self.input = activation_fn(self.input)
                self.input = tf.nn.conv2d(self.input, w, strides=[1, 1, 1, 1], padding='SAME')
                b = self.const_variable(name='bias', shape=[output_channels], value=0.0)
                self.input = tf.add(self.input, b)

            # Add input and output for final return
            self.input = self.input + additive_output
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def noisy_and(self, num_classes):
        """ Multiple Instance Learning (MIL), flexible pooling function
        :param num_classes: int, determine number of output maps
        """
        assert self.input.get_shape()[3] == num_classes  # input tensor should have map depth equal to # of classes
        scope = 'noisyAND'
        with tf.variable_scope(scope):
            a = self.const_variable(name='a', shape=[1], value=1.0)
            b = self.const_variable(name='b', shape=[1, num_classes], value=0.0)
            mean = tf.reduce_mean(self.input, axis=[1, 2])
            self.input = (tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)) / (
                        tf.sigmoid(a * (1 - b)) - tf.sigmoid(-a * b))
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def get_output(self):
        """
        :return tf.Tensor, output of network
        """
        return self.input

    def batch_norm(self, x, type='conv', epsilon=1e-3):
        """
        Batch Normalization: Apply mean subtraction and variance scaling
        :param x: input feature map stack
        :param type: string, either 'conv' or 'fc'
        :param epsilon: float
        :return: output feature map stack
        """
        # Determine indices over which to calculate moments, based on layer type
        if type == 'conv':
            size = [0, 1, 2]
        else:  # type == 'fc'
            size = [0]

        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(x, size, keep_dims=True)

        # Apply the initial batch normalizing transform
        z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        return z1_hat

    @staticmethod
    def print_log(message):
        """ Writes a message to terminal screen and logging file, if applicable"""
        print(message)
        logging.info(message)

    @staticmethod
    def weight_variable(name, shape):
        """
        :param name: string
        :param shape: 4D array
        :return: tf variable
        """
        w = tf.get_variable(name=name, shape=shape, initializer=init.variance_scaling_initializer())
        weights_norm = tf.reduce_sum(tf.nn.l2_loss(w), name=name + '_norm')  # Should user want to optimize weight decay
        tf.add_to_collection('weight_losses', weights_norm)
        return w

    @staticmethod
    def const_variable(name, shape, value):
        """
        :param name: string
        :param shape: 1D array
        :param value: float
        :return: tf variable
        """
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))


class Data:
    """
    A Class to handle data I/O and batching in TensorFlow.
    Use class methods for datasets:
        - That can be loaded into memory all at once.
        - That use the placeholder function in TensorFlow
    Use batch_inputs method et al for datasets:
        - That can't be loaded into memory all at once.
        - That use queueing and threading fuctions in TesnorFlow
    """

    def __init__(self, flags, valid_percent=0.2, test_percent=0.15):
        self.flags = flags
        train_images, train_labels, self.test_images, self.test_labels = self.load_data(test_percent)
        self._num_test_images = len(self.test_labels)
        self._num_train_images = math.floor(len(train_labels) * (1 - valid_percent))
        self._num_valid_images = len(train_labels) - self._num_train_images
        self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
            self.split_data(train_images, train_labels)

        self.train_epochs_completed = 0
        self.index_in_train_epoch = 0
        self.index_in_valid_epoch = 0
        self.index_in_test_epoch = 0

    def load_data(self, test_percent=0.15):
        """Load the dataset into memory. If data is not divided into train/test, use test_percent to divide the data"""
        train_images = list()
        train_labels = list()
        test_images = list()
        test_labels = list()
        return train_images, train_labels, test_images, test_labels

    def split_data(self, train_images, train_labels):
        """
        :param train_images: numpy array (image_dim, image_dim, num_images)
        :param train_labels: numpy array (labels)
        :return: train_images, train_labels, valid_images, valid_labels
        """
        valid_images = train_images[:self.num_valid_images]
        valid_labels = train_labels[:self.num_valid_images]
        train_images = train_images[self.num_valid_images:]
        train_labels = train_labels[self.num_valid_images:]
        return train_images, train_labels, valid_images, valid_labels

    def next_train_batch(self, batch_size):
        """
        Return the next batch of examples from train data set
        :param batch_size: int, size of image batch returned
        :return train_labels: list, of labels
        :return images: list, of images
        """
        start = self.index_in_train_epoch
        self.index_in_train_epoch += batch_size
        if self.index_in_train_epoch > self.num_train_images:
            # Finished epoch
            self.train_epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self.num_train_images)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]

            # Start next epoch
            start = 0
            self.index_in_train_epoch = batch_size
            assert batch_size <= self.num_train_images

        end = self.index_in_train_epoch
        return self.train_labels[start:end], self.img_norm(self.train_images[start:end])

    def next_valid_batch(self, batch_size):
        """
        Return the next batch of examples from validiation data set
        :param batch_size: int, size of image batch returned
        :return train_labels: list, of labels
        :return images: list, of images
        """
        start = self.index_in_valid_epoch
        if self.index_in_valid_epoch + batch_size > self.num_valid_images:
            batch_size = 1
        self.index_in_valid_epoch += batch_size
        end = self.index_in_valid_epoch
        return self.valid_labels[start:end], self.img_norm(self.valid_images[start:end]), end, batch_size

    def next_test_batch(self, batch_size):
        """
        Return the next batch of examples from test data set
        :param batch_size: int, size of image batch returned
        :return train_labels: list, of labels
        :return images: list, of images
        """
        start = self.index_in_test_epoch
        print(start)
        if self.index_in_test_epoch + batch_size > self.num_test_images:
            batch_size = 1
        self.index_in_test_epoch += batch_size
        end = self.index_in_test_epoch
        return self.test_labels[start:end], self.img_norm(self.test_images[start:end]), end, batch_size

    @property
    def num_train_images(self):
        return self._num_train_images

    @property
    def num_test_images(self):
        return self._num_test_images

    @property
    def num_valid_images(self):
        return self._num_valid_images

    @staticmethod
    def img_norm(x, max_val=255):
        """
        Normalizes stack of images
        :param x: input feature map stack, assume uint8
        :param max_val: int, maximum value of input tensor
        :return: output feature map stack
        """
        return (x * (1 / max_val) - 0.5) * 2  # returns scaled input ranging from [-1, 1]

    @classmethod
    def batch_inputs(cls, read_and_decode_fn, tf_file, batch_size, mode="train", num_readers=4, num_threads=4,
                     min_examples=5000):
        with tf.name_scope('batch_processing'):
            if mode == "train":
                epochs = None
            else:  # test or validate or eval
                epochs = 1
            example_serialized = cls.queue_setup(tf_file, epochs, batch_size, num_readers, min_examples)
            decoded_data = cls.thread_setup(read_and_decode_fn, example_serialized, num_threads)
            return tf.train.batch_join(decoded_data, batch_size=batch_size)

    @staticmethod
    def queue_setup(filename, epochs, batch_size, num_readers, min_examples):
        """ Sets up the queue runners for data input """
        filename_queue = tf.train.string_input_producer([filename], num_epochs=epochs, shuffle=True, capacity=16)
        examples_queue = tf.RandomShuffleQueue(capacity=min_examples + 3 * batch_size,
                                               min_after_dequeue=min_examples, dtypes=[tf.string])
        enqueue_ops = list()
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
        return example_serialized

    @staticmethod
    def thread_setup(read_and_decode_fn, example_serialized, num_threads):
        """ Sets up the threads within each reader """
        decoded_data = list()
        for _ in range(num_threads):
            decoded_data.append(read_and_decode_fn(example_serialized))
        return decoded_data

    @staticmethod
    def init_threads(tf_session):
        """ Starts threads running """
        coord = tf.train.Coordinator()
        threads = list()
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(tf_session, coord=coord, daemon=True, start=True))
        return threads, coord

    @staticmethod
    def exit_threads(threads, coord):
        """ Closes out all threads """
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


class Model:
    """
    A Class for easy Model Training.
    Methods:
        See list in __init__() function
    """

    def __init__(self, flags, run_num, vram=0.25):
        print(flags)

        # Define constants
        self.global_step = 0
        self.step = 0
        self.num_test_images = 0
        self.num_valid_images = 0
        self.num_train_images = 0
        self.run_num = run_num

        # Define other elements
        self.results = list()
        self.flags = flags

        # Run initialization functions
        self._check_file_io(run_num)
        self._data()
        self._set_seed()
        self._network()
        self._optimizer()
        self._summaries()
        self.merged, self.saver, self.sess, self.writer = self._set_tf_functions(vram)
        self._initialize_model()

    def __enter__(self):
        return self

    def __exit__(self, *err):
        self.close()

    def _data(self):
        """Define data"""
        raise NotImplementedError

    def _network(self):
        """Define network"""
        raise NotImplementedError

    def _optimizer(self):
        """Define optimizer"""
        raise NotImplementedError

    def _check_file_io(self, run_num):
        folder = 'Model' + str(run_num) + '/'
        self.flags['restore_directory'] = self.flags['save_directory'] + self.flags['model_directory'] + folder
        self.make_directory(self.flags['restore_directory'])
        logging.basicConfig(filename=self.flags['restore_directory'] + 'ModelInformation.log', level=logging.INFO)

    def _set_seed(self):
        tf.set_random_seed(self.flags['seed'])
        np.random.seed(self.flags['seed'])
        self.print_log("Seed: %d" % self.flags['seed'])

    def _summaries(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
            print(var.name)

    def _set_tf_functions(self, vram=0.25):
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = vram
        sess = tf.InteractiveSession(config=config)
        writer = tf.summary.FileWriter(self.flags['restore_directory'], sess.graph)
        return merged, saver, sess, writer

    def _restore(self):
        new_saver = tf.train.import_meta_graph('./' + self.flags['restore_directory'] + self.flags['restore_file'])
        new_saver.restore(self.sess, tf.train.latest_checkpoint('./' + self.flags['restore_directory']))
        self.print_log("Model restored from %s" % self.flags['restore_file'])

    def _setup_metrics(self):
        self.print_log('Date: ' + str(datetime.datetime.now()).split('.')[0])
        self.print_log('Batch_size: ' + self.check_str(self.flags['batch_size']))
        self.print_log('Model: ' + self.check_str(self.flags['model_directory']))

    def _initialize_model(self):
        self._setup_metrics()
        self.sess.run(tf.local_variables_initializer())
        if self.flags['restore'] is True:
            self._restore()
        else:
            self.sess.run(tf.global_variables_initializer())
            self.print_log("Model training from scratch.")

    def _save_model(self, section):
        self.print_log("Optimization Finished!")
        checkpoint_name = self.flags['restore_directory'] + 'part_%d' % section + '.ckpt'
        save_path = self.saver.save(self.sess, checkpoint_name)
        self.print_log("Model saved in file: %s" % save_path)

    def _record_training_step(self, summary):
        self.writer.add_summary(summary=summary, global_step=self.global_step)
        self.step += 1
        self.global_step += 1

    @staticmethod
    def _check_flags(flags):
        flags_keys = ['restore', 'restore_file', 'batch_size', 'display_step', 'weight_decay', 'lr_decay',
                      'lr_iters']
        for k in flags_keys:
            try:
                flags[k]
            except KeyError:
                print('The key %s is not defined in the flags dictionary. Please define and run again' % k)
        return flags

    @staticmethod
    def make_directory(folder_path):
        """ Make directory at folder_path if it does not exist """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def print_log(message):
        """ Print message to terminal and to logging document if applicable """
        print(message)
        logging.info(message)

    @staticmethod
    def check_str(obj):
        """ Returns a string for various input types """
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)
