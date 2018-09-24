from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt

class WaveFile(object):
    def __init__(self, dir):
        self.hz = self.c1 = self.c2 = self.length = None
        self.dir = dir
        self.read(dir)

    def read(self, file):
        wave_file = wavfile.read(file)
        self.c1 = wave_file[1][:, 0]
        self.c2 = wave_file[1][:, 1]
        self.hz = wave_file[0]
        self.length = len(self.c1) / wave_file[0]

    def generate(self, file, n):

        for _ in range(n):
            self.read()


    def std(self):
        return np.std(self.c1)

    def mean(self):
        return np.mean(self.c1)

    def __repr__(self):
        return "Sampling rate: {0} Hz\nLength: {1:.2f} s\nDir: {2}".format(self.hz, self.length, self.dir)


if __name__ == '__main__':
    wave_file = WaveFile(dir="heartbeat.wav")
    print(wave_file)

    print(wave_file.std())

    plt.subplot(3, 1, 1)
    plt.hist(wave_file.c1, range=(-5000, 5000), bins=500, density=False, label="wave")
    plt.legend()


    import matplotlib.mlab as mlab
    import math

    plt.subplot(3, 1, 2)
    mu = wave_file.mean()
    sigma = wave_file.std()
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    plt.plot(x, mlab.normpdf(x, mu, sigma))

    plt.plot(wave_file.c1)
    #plt.show()
