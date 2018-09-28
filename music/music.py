from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt

class WaveFile(object):
    def __init__(self, dir):
        self.hz = self.c1 = self.c2 = self.audio = self.length = None
        self.dir = dir
        if dir is not None:
            self.read(dir)

    def read(self, file):
        wave_file = wavfile.read(file)
        self.c1 = wave_file[1]
        self.audio = (self.c1 - self.mean(self.c1)) / self.std(self.c1)
        self.hz = wave_file[0]
        self.length = len(self.c1) / wave_file[0]

    def audio(self):
        """Returns numpy array of audio mapped from -1 to 1"""
        return self.audio / 32767

    def generate_samples(self, intervals, samples):
        """
        Split .wav audio file into intervals, calculate mean/std, draws n samples from distribution.
        :param intervals: intervals per second of audio file
        :param samples: number of samples from normal distribution
        :return: 2D-Array, Shape(intervals, samples)
        """
        divided_audio = np.array_split(self.audio, int(intervals * self.length))
        generated_samples = []

        for part in divided_audio:
            sample = np.random.normal(self.mean(part), self.std(part), samples)
            generated_samples.append(sample)

        return generated_samples

    def std(self, x):
        return np.std(x)

    def mean(self, x):
        return np.mean(x)

    def plot(self):
        """Plot four plots: histogram, normal distribution, spectogram and wave"""
        import matplotlib.mlab as mlab
        plt.subplot(4, 1, 1)
        plt.hist(wave_file.c1, range=(-5000, 5000), bins=500, density=False, label="wave")
        plt.legend()

        plt.subplot(4, 1, 2)
        mu = wave_file.mean(wave_file.c1)
        sigma = wave_file.std(wave_file.c1)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
        plt.plot(x, mlab.normpdf(x, mu, sigma))

        plt.subplot(4, 1, 3)
        plt.specgram(wave_file.c1, Fs=wave_file.hz)

        plt.subplot(4, 1, 4)
        plt.plot(wave_file.c1)
        plt.show()

    def __repr__(self):
        return "Sampling rate: {0} Hz\nLength: {1:.2f} s\nDir: {2}".format(self.hz, self.length, self.dir)


if __name__ == '__main__':
    wave_file = WaveFile(dir="sweep_200hz-500hz.wav")
    print(wave_file)

    samples_x = wave_file.generate_samples(intervals=10, samples=20)