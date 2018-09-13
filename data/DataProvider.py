from tensorflow.examples.tutorials.mnist import input_data


class DataProvider:
    @staticmethod
    def get_mnist():
        return input_data.read_data_sets("MNIST_data/", one_hot=True), 28

    @staticmethod
    def get_celeb_a():
        # todo Implement this function so it returns the same format as the get_mnist
        # currently we don't need test and validation I guess
        raise NotImplementedError


if __name__ == '__main__':
    DataProvider.get_mnist()
