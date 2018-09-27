import numpy as np
import os
from abc import ABC, abstractmethod

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.config import TRAINING, TF_BOARD, IMAGES


class VAE(ABC):

    def __init__(self, data_set, hidden_size, batch_size, learning_rate, max_epochs):
        self.data_set, self.width = data_set()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.iterations_per_epoch = self.data_set.train.num_examples // self.batch_size

        # *** FEED FORWARD
        # encoder
        with tf.variable_scope("input"):
            self.images_1d = tf.placeholder(tf.float32, [None, self.width ** 2])
            self.dynamic_batch_size = tf.shape(self.images_1d)[0]
            images_2d = tf.reshape(self.images_1d, [self.dynamic_batch_size, self.width, self.width, 1])

        with tf.variable_scope('encode'):
            z_mean, z_stddev = self.encode(images_2d)

        # generate samples
        with tf.variable_scope('latent_space'):
            samples = tf.random_normal([self.dynamic_batch_size, self.hidden_size], 0, 1, dtype=tf.float32,
                                       name='samples')
            guessed_z = z_mean + (z_stddev * samples)

            # real z as input, needs to be separate for feeding in batches with different sizes
            self.latent_z = guessed_z

        with tf.variable_scope('decode'):
            self.generated_images = self.decode(self.latent_z)
            generated_flat = tf.reshape(self.generated_images, [self.dynamic_batch_size, -1])

        with tf.variable_scope('loss'):
            # pixel correspondence loss (input/output)
            self.generation_loss = tf.abs(tf.reduce_sum(tf.square(self.images_1d - generated_flat))) / tf.to_float(
                self.dynamic_batch_size)

            # KL-divergence
            self.latent_loss = 0.5 * tf.reduce_sum(
                tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)

            # total loss
            self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)

        # *** OPTIMIZER
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()

        # loss
        tf.summary.scalar("total", self.cost, family="loss")
        tf.summary.scalar("generation", tf.reduce_mean(self.generation_loss), family="loss")
        tf.summary.scalar("KL-Divergence", tf.reduce_mean(self.latent_loss), family="loss")
        self.merged_summary_op = tf.summary.merge_all()
        self.image_summary = tf.summary.image("image", tf.reshape(self.generated_images,
                                                                  (self.dynamic_batch_size, self.width, self.width,
                                                                   1)))
        os.environ["Model"] = self.__class__.__name__

    @abstractmethod
    def encode(self, input_images):
        raise NotImplementedError

    @abstractmethod
    def decode(self, guessed_z):
        raise NotImplementedError

    def create_folders(self):
        os.makedirs(TRAINING(), exist_ok=True)
        os.makedirs(TF_BOARD(), exist_ok=True)
        os.makedirs(IMAGES(), exist_ok=True)

    def train(self):
        self.create_folders()

        static_batch = self.data_set.test.next_batch(
            self.batch_size)[0]
        summary_writer = tf.summary.FileWriter(TF_BOARD(), graph=tf.get_default_graph())

        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(self.init)
            pgbar = tqdm(range(self.max_epochs))
            for epoch in pgbar:
                gen_loss = None
                lat_loss = None

                for idx in range(self.iterations_per_epoch):
                    batch = self.data_set.train.next_batch(self.batch_size)[0]
                    _, gen_loss, lat_loss, summary = sess.run(
                        (self.optimizer, self.generation_loss, self.latent_loss, self.merged_summary_op),
                        feed_dict={self.images_1d: batch})

                    # Write logs at every iteration
                    summary_writer.add_summary(summary, epoch * self.iterations_per_epoch + idx)

                # todo get these values from last summary
                tqdm.write("epoch {}: genloss {} latloss {}".format(epoch, np.mean(gen_loss), np.mean(lat_loss)))

                # save current model
                saver.save(sess, TRAINING() + "/train", global_step=epoch * self.data_set.train.num_examples)

                # log images
                generated_images, images_summary = sess.run((self.generated_images, self.image_summary),
                                                            feed_dict={self.images_1d: static_batch})
                summary_writer.add_summary(images_summary,
                                           global_step=epoch * self.data_set.train.num_examples)

    def load_pretrained(self, path):
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(path))
            # you can change the 1 to a different number, it indicates how many images are generated
            number_of_images = 100
            z = np.random.standard_normal((number_of_images, self.hidden_size))
            generated_image = sess.run(self.generated_images,
                                       feed_dict={self.dynamic_batch_size: number_of_images, self.latent_z: z})
            # plot somehow only works with two dimension (the third is 1 anyway)
            img = generated_image[0, :, :, 0]
            np.save("images.npy", generated_image)
            plt.plot(img)
            plt.show()
            plt.imsave("./sample.png", img)


class SimpleVAE(VAE):
    def encode(self, input_images):
        # 28x28x1 -> 14x14x16
        conv_1 = tf.layers.conv2d(input_images, 16, kernel_size=5, activation=tf.nn.leaky_relu, strides=2,
                                  padding='same', name='d_conv_0')
        # 14x14x16 -> 7x7x32
        conv_2 = tf.layers.conv2d(conv_1, 32, kernel_size=5, activation=tf.nn.leaky_relu, strides=2, padding='same',
                                  name='d_conv_1')
        # 7x7x32 -> (7*7*32)
        flatten = tf.reshape(conv_2, [self.dynamic_batch_size, 7 * 7 * 32])

        mean = tf.layers.dense(flatten, self.hidden_size, name='fc_mean')
        stddev = tf.layers.dense(flatten, self.hidden_size, name='fc_std')
        return mean, stddev

    def decode(self, z):
        starting_res_1d = tf.layers.dense(z, 7 * 7 * 32, activation=tf.nn.leaky_relu, name='fc_upscale')
        starting_res_2d = tf.reshape(starting_res_1d, [self.dynamic_batch_size, 7, 7, 32])
        conv_1T = tf.layers.conv2d_transpose(starting_res_2d, 16, kernel_size=5, activation=tf.nn.leaky_relu,
                                             strides=2,
                                             padding='same', name='u_conv_0')
        conv_2T = tf.layers.conv2d_transpose(conv_1T, 1, kernel_size=5, activation=tf.nn.leaky_relu,
                                             strides=2, padding='same', name='u_conv_1')
        sig = tf.nn.sigmoid(conv_2T, name='sigmoid')
        return sig
