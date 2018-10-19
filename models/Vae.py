import numpy as np
import os
import datetime
from abc import ABC, abstractmethod

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.config import TRAINING, TF_BOARD, IMAGES


class VAE(ABC):

    def __init__(self, data_set, hidden_size, batch_size, learning_rate, max_epochs):
        self.data_set, self.width = data_set(batch_size)  # self.data_set is now a tf_records_dataset
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs

        # *** FEED FORWARD
        # encoder
        with tf.variable_scope("input"):
            self.images_1d = tf.placeholder(tf.float32, [None, self.width * self.width * 3])
            self.dynamic_batch_size = tf.shape(self.images_1d)[0]
            images_2d = tf.reshape(self.images_1d, [self.dynamic_batch_size, 3, self.width, self.width])
            # todo make sure that the images are in right order! i couldnt test if this works
            images_2d = tf.transpose(images_2d, perm=[0, 2, 3, 1])

        with tf.variable_scope('encode'):
            z_mean, z_stddev = self.encode(images_2d)

        # generate samples
        with tf.variable_scope('latent_space'):
            samples = tf.random_normal([self.dynamic_batch_size, self.hidden_size], 0, 1, dtype=tf.float32,
                                       name='samples')
            guessed_z = z_mean + tf.sqrt(tf.exp(z_stddev)) * samples

            # real z as input, needs to be separate for feeding in batches with different sizes
            self.latent_z = guessed_z

        with tf.variable_scope('decode'):
            self.generated_images = self.decode(self.latent_z)

        with tf.variable_scope('loss'):
            # pixel correspondence loss (input/output)
            epsilon = 1e-10
            recon_loss = -tf.reduce_sum(
                images_2d * tf.log(epsilon + self.generated_images) + (1 - images_2d) * tf.log(
                    epsilon + 1 - self.generated_images))
            self.generation_loss = tf.reduce_mean(recon_loss) / tf.to_float(self.dynamic_batch_size)
            # todo maybe remove the devision
            latent_loss = -0.5 * tf.reduce_sum(
                1 + z_stddev - tf.square(z_mean) - tf.exp(z_stddev), axis=1)
            self.latent_loss = tf.reduce_mean(latent_loss)

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
                                                                   3)))
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

        summary_writer = tf.summary.FileWriter(TF_BOARD(), graph=tf.get_default_graph())

        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(self.init)
            # get a random batch as static batch
            static_batch = sess.run(self.data_set.make_one_shot_iterator().get_next())

            # thing for printing progress
            pgbar = tqdm(range(self.max_epochs))
            # current numbers of batches
            iteration = 0

            # log images in beginning as well
            generated_images, images_summary = sess.run((self.generated_images, self.image_summary),
                                                        feed_dict={self.images_1d: static_batch})
            summary_writer.add_summary(images_summary,
                                       global_step=iteration)

            for epoch in pgbar:
                # create iterator for getting the data
                tfrecord_iterator = self.data_set.make_one_shot_iterator()
                batch = tfrecord_iterator.get_next()

                gen_loss = None
                lat_loss = None

                # while we get data from the iterator train
                # if the tf.errors.OutOfRangeError is thrown the epoch is done
                while True:
                    try:
                        _, gen_loss, lat_loss, summary = sess.run(
                            (self.optimizer, self.generation_loss, self.latent_loss, self.merged_summary_op),
                            feed_dict={self.images_1d: batch.eval()})  # i don't like the .eval() because it converts
                        # the tensor back to numpy, but in theory we could use the tensor directly. but we can't feed it
                        # here. So maybe there is an option to pass the iterator instead?

                        # Write logs at every iteration
                        summary_writer.add_summary(summary, iteration)
                        iteration += 1
                    except tf.errors.OutOfRangeError:
                        break
                # todo get these values from last summary
                tqdm.write("epoch {}: genloss {} latloss {}".format(epoch, np.mean(gen_loss), np.mean(lat_loss)))

                # save current model
                saver.save(sess, TRAINING() + "/train", global_step=iteration)

                # log images
                generated_images, images_summary = sess.run((self.generated_images, self.image_summary),
                                                            feed_dict={self.images_1d: static_batch})
                summary_writer.add_summary(images_summary,
                                           global_step=iteration)

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

    def generate_images(self, weights, dists, save=False):
        """
        Generate images, using specified network weights, from distribution from audio file.
        :param weights: path to network weights
        :param dists: sampled distributions
        :param save: true or false
        :return: numpy array of images
        """
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(weights))
            images = sess.run(self.generated_images,

                              feed_dict={self.dynamic_batch_size: len(dists), self.latent_z: dists})
            if save:
                time = datetime.datetime.now().strftime("./generated_images/%d-%m-%y %H-%M-%S")
                os.makedirs(time, exist_ok=True)
                for i, image in enumerate(images):
                    plt.imsave("{}/image_{}.png".format(time, i), image[:, :, 0])

            return images


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

        mean = tf.layers.dense(flatten, self.hidden_size, name='fc_mean', activation=None)
        stddev = tf.layers.dense(flatten, self.hidden_size, name='fc_std', activation=None)
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


class CelebAVAE(VAE):
    def encode(self, input_images):
        # 128x128x1 -> 64x64x16
        conv_1 = tf.layers.conv2d(input_images, 16, kernel_size=3, activation=tf.nn.leaky_relu, strides=2,
                                  padding='same', name='d_conv_0')
        # 64x64x16 -> 32x32x32
        conv_2 = tf.layers.conv2d(conv_1, 32, kernel_size=3, activation=tf.nn.leaky_relu, strides=2, padding='same',
                                  name='d_conv_1')
        # 32x32x32 -> 16x16x64
        conv_3 = tf.layers.conv2d(conv_2, 64, kernel_size=3, activation=tf.nn.leaky_relu, strides=2, padding='same',
                                  name='d_conv_2')
        # 16x16x64 -> 8x8x128
        conv_4 = tf.layers.conv2d(conv_3, 128, kernel_size=3, activation=tf.nn.leaky_relu, strides=2, padding='same',
                                  name='d_conv_3')
        # 8x8x128 -> 4x4x256
        conv_5 = tf.layers.conv2d(conv_4, 256, kernel_size=3, activation=tf.nn.leaky_relu, strides=2, padding='same',
                                  name='d_conv_4')
        # 4x4x256 -> 2x2x512
        conv_6 = tf.layers.conv2d(conv_5, 512, kernel_size=3, activation=tf.nn.leaky_relu, strides=2, padding='same',
                                  name='d_conv_5')
        # 2x2x512 -> (2*2*512)
        flatten = tf.reshape(conv_6, [self.dynamic_batch_size, 2*2*512])

        mean = tf.layers.dense(flatten, self.hidden_size, name='fc_mean', activation=None)
        stddev = tf.layers.dense(flatten, self.hidden_size, name='fc_std', activation=None)
        return mean, stddev

    def decode(self, z):
        starting_res_1d = tf.layers.dense(z, 2*2*512, activation=tf.nn.leaky_relu, name='fc_upscale')
        starting_res_2d = tf.reshape(starting_res_1d, [self.dynamic_batch_size, 2, 2, 512])

        conv_1T = tf.layers.conv2d_transpose(starting_res_2d, 256, kernel_size=3, activation=tf.nn.leaky_relu,
                                             strides=2,
                                             padding='same', name='u_conv_0')
        conv_2T = tf.layers.conv2d_transpose(conv_1T, 128, kernel_size=3, activation=tf.nn.leaky_relu,
                                             strides=2, padding='same', name='u_conv_1')
        conv_3T = tf.layers.conv2d_transpose(conv_2T, 64, kernel_size=3, activation=tf.nn.leaky_relu,
                                             strides=2, padding='same', name='u_conv_2')
        conv_4T = tf.layers.conv2d_transpose(conv_3T, 32, kernel_size=3, activation=tf.nn.leaky_relu,
                                             strides=2, padding='same', name='u_conv_3')
        conv_5T = tf.layers.conv2d_transpose(conv_4T, 16, kernel_size=3, activation=tf.nn.leaky_relu,
                                             strides=2, padding='same', name='u_conv_4')
        conv_6T = tf.layers.conv2d_transpose(conv_5T, 3, kernel_size=3, activation=tf.nn.leaky_relu,
                                             strides=2, padding='same', name='u_conv_5')
        sig = tf.nn.sigmoid(conv_6T, name='sigmoid')
        return sig
