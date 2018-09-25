import datetime
import os
from collections import namedtuple

from data.DataProvider import DataProvider

model_args = namedtuple("model_args", "data_set, hidden_size, batch_size, learning_rate, max_epochs")

DEFAULT = model_args(data_set=DataProvider.get_mnist, hidden_size=20, batch_size=128, learning_rate=.001, max_epochs=10)

TIME = str(datetime.datetime.now())

TRAINING = lambda: "./training/{}/{}".format(os.environ["Model"], TIME)
TF_BOARD = lambda: TRAINING()
IMAGES = lambda: TRAINING() + "/images"
