from scipy.io import wavfile
import numpy as np
import librosa


class WaveFile(object):
    def __init__(self, dir):
        self.hz = self.c1 = self.c2 = self.audio = self.length = None
        self.dir = dir
        if dir is not None:
            self.read(dir)

    def read(self, file):
        self.hz, self.c1 = wavfile.read(file)
        # if there is only one channel just repeat it to simulate stereo
        if len(self.c1.shape) == 1:
            self.c1 = np.column_stack((self.c1, self.c1))
        if not len(self.c1.shape) == 2:
            raise NotImplementedError("Files with more than 2 channels are not supported at the moment!")
        # maybe we dont want to normalize across both dimensions but for now its fine
        self.audio = (self.c1 - self.mean(self.c1)) / self.std(self.c1)
        self.length = len(self.c1) / self.hz

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
            #sample = np.random.normal(self.mean(part), self.std(part), 1)
            #sample = np.repeat(sample, samples)
            generated_samples.append(sample)

        return generated_samples

    def generate_mell(self, intervals, samples):
        """
        Split .wav audio file into intervals, calculate mean/std, draws n samples from distribution.
        :param intervals: intervals per second of audio file
        :param samples: number of samples from normal distribution
        :return: 2D-Array, Shape(intervals, samples)
        """

        y, sr = librosa.load(self.dir)

        D = np.abs(librosa.stft(y)) ** 2
        S = librosa.feature.melspectrogram(S=D)

        # Passing through arguments to the Mel filters
        S2 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(sr / intervals),
                                            hop_length=int(sr / intervals), n_mels=samples,
                                            )
        S_dB = librosa.power_to_db(S2, ref=np.max)
        S_dB += 80
        S_dB /= S_dB.max()

        return S_dB.T * 2 - 1


    def std(self, x):
        return np.std(x)

    def mean(self, x):
        return np.mean(x)

    def plot(self):
        """Plot four plots: histogram, normal distribution, spectogram and wave"""
        import matplotlib.mlab as mlab

        """
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
        """
        #plt.subplot(4, 1, 4)

    def __repr__(self):
        return "Sampling rate: {0} Hz\nLength: {1:.2f} s\nDir: {2}".format(self.hz, self.length, self.dir)


if __name__ == '__main__':
    wave_file = WaveFile(dir="heartbeat.wav")
    print(wave_file)
    #wave_file = WaveFile(dir="we_will_rock_you.wav")
    #print(wave_file)
    #wave_file = WaveFile(dir="frequency_test.wav")
    #print(wave_file)
    # print(wave_file)

    samples_x = wave_file.generate_mell(intervals=24, samples=5)

    #wave_file.plot()
    #half_sec = wave_file.audio[0:int(wave_file.hz / 2), 0]

    #plt.plot(np.linspace(0, 500, int(wave_file.hz / 2), endpoint=True), half_sec, label="Heartbeat audio signal")
    #plt.specgram(samples_x)
    #plt.xlabel("Time (ms)")
    #plt.ylabel("Amplitude")
    #plt.legend()
    #plt.show()