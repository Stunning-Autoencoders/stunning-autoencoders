import time

import numpy as np
from matplotlib import pyplot
from tensorflow.examples.tutorials.mnist import input_data
import os

import tensorflow as tf

TRAINING = "./training/" + str(time.time())
TF_BOARD = TRAINING
IMAGES = TRAINING + "/images"


# this code is mainly copied from:
# https://github.com/kvfrans/variational-autoencoder
class LatentAttention():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100

        # *** FEED FORWARD
        # encoder
        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])
        z_mean, z_stddev = self.encode(image_matrix)

        # generate samples
        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        # decoder
        self.generated_images = self.decode(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28 * 28])

        # *** LOSS
        # one 2 one loss (input/output)
        # self.generation_loss = -tf.reduce_sum(
        #    self.images * tf.log(1e-8 + generated_flat) + (1 - self.images) * tf.log(1e-8 + 1 - generated_flat), 1)
        self.generation_loss = tf.abs(tf.reduce_sum(tf.square(self.images - generated_flat))) / self.batchsize

        # KL-divergence
        self.latent_loss = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)

        # total loss
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)

        # *** OPTIMIZER
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.init = tf.global_variables_initializer()

        # *** SUMMARIES
        tf.summary.scalar("total", self.cost, family="loss")
        tf.summary.scalar("generation", tf.reduce_mean(self.generation_loss), family="loss")
        tf.summary.scalar("KL-Divergence", tf.reduce_mean(self.latent_loss), family="loss")
        self.merged_summary_op = tf.summary.merge_all()

    # encoder
    def encode(self, input_images):
        with tf.variable_scope("encode"):
            # 28x28x1 -> 14x14x16
            h1 = tf.layers.conv2d(input_images, 16, kernel_size=5, activation=tf.nn.leaky_relu, strides=2,
                                  padding='same')
            # 14x14x16 -> 7x7x32
            h2 = tf.layers.conv2d(h1, 32, kernel_size=5, activation=tf.nn.leaky_relu, strides=2, padding='same')
            # 7x7x32 -> (7*7*32)
            h2_flat = tf.reshape(h2, [self.batchsize, 7 * 7 * 32])

            w_mean = tf.layers.dense(h2_flat, self.n_z)
            w_stddev = tf.layers.dense(h2_flat, self.n_z)
        return w_mean, w_stddev

    # decoder
    def decode(self, z):
        with tf.variable_scope("decode"):
            z_develop = tf.layers.dense(z, 7 * 7 * 32, activation=tf.nn.leaky_relu)
            z_matrix = tf.reshape(z_develop, [self.batchsize, 7, 7, 32])
            h1 = tf.layers.conv2d_transpose(z_matrix, 16, kernel_size=5, activation=tf.nn.leaky_relu, strides=2,
                                            padding='same')
            h2 = tf.layers.conv2d_transpose(h1, 1, kernel_size=5, activation=tf.nn.leaky_relu,
                                            strides=2, padding='same')
            h2 = tf.nn.sigmoid(h2)

        return h2

    def train(self):
        os.makedirs(TRAINING, exist_ok=True)
        os.makedirs(TF_BOARD, exist_ok=True)
        os.makedirs(IMAGES, exist_ok=True)
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        summary_writer = tf.summary.FileWriter(TF_BOARD, graph=tf.get_default_graph())

        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(100):
                gen_loss = None
                lat_loss = None

                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss, summary = sess.run(
                        (self.optimizer, self.generation_loss, self.latent_loss, self.merged_summary_op),
                        feed_dict={self.images: batch})

                    # Write logs at every iteration
                    summary_writer.add_summary(summary, epoch * (self.n_samples / self.batchsize) + idx)

                print("epoch {}: genloss {} latloss {}".format(epoch, np.mean(gen_loss), np.mean(lat_loss)))
                saver.save(sess, TRAINING + "/train", global_step=epoch * self.n_samples)
                generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                generated_test = generated_test.reshape(self.batchsize, 28, 28)
                pyplot.imsave(IMAGES + "/{0:03d}.jpg".format(epoch), generated_test[0])


if __name__ == '__main__':
    model = LatentAttention()
    model.train()
