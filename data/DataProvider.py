from tensorflow.examples.tutorials.mnist import input_data
from data.celebA import read_data_sets

class DataProvider:
    @staticmethod
    def get_mnist():
        print('dataset output',input_data.read_data_sets("MNIST_data/", one_hot=True))
        return input_data.read_data_sets("MNIST_data/", one_hot=True), 28

    @staticmethod
    def get_celeb_a():
        # todo Implement this function so it returns the same format as the get_mnist
        # currently we don't need test and validation I guess
        return read_data_sets("celebA_data/"), 128


if __name__ == '__main__':
    #DataProvider.get_mnist()
    DataProvider.get_celeb_a()
