import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

file = "sweep_200hz-500hz.wav"
#file = "sweep_200Hz_1000Hz_-3dBFS_1s.wav"
#file = "heartbeat.wav"
#file = "we_will_rock_you.wav"

y, sr = librosa.load(file)
#librosa.feature.melspectrogram(y=y, sr=sr)

# Using a pre-computed power spectrogram

D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)

# Passing through arguments to the Mel filters
S2 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(sr/24), hop_length=int(sr/24),  n_mels=20,
                                    )

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(2,1,1)

librosa.display.specshow(librosa.power_to_db(S2, ref=np.max), y_axis='mel', x_axis='time')
#plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram (Heartbeat.wav)')
plt.tight_layout()

#plt.figure(figsize=(10, 4))
#librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis="mel", x_axis="time")
#plt.colorbar(format='%+2.0f dB')
#plt.title('Regular spectrogram')
#plt.tight_layout()

ax2 = plt.subplot(212)
plt.plot(y)
plt.title('Heartbeat.wav')
plt.tight_layout()
plt.xlabel("Samples (22055 samples per second)")
plt.ylabel("Amplitude")