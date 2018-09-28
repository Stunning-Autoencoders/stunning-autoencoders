from config.config import DEFAULT
from models.Vae import SimpleVAE
from music.music import WaveFile
from utils.utils import create_video
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # SimpleVAE(*DEFAULT).train()
    vae = SimpleVAE(*DEFAULT)
    # vae.load_pretrained("./training/SimpleVAE/run3")

    wave_file = WaveFile(dir="music/sweep_200hz-500hz.wav")
    sampled_dists = wave_file.generate_samples(intervals=2, samples=20)
    images = vae.generate_images(weights="./training/SimpleVAE/run3",
                                            dists=sampled_dists,
                                            save=True)

    images = (images[:, :, :, 0] * 255.999).astype(np.uint8)
    images = np.stack((images, images, images), -1)
    images = [cv2.resize(i, dsize=(320, 320)) for i in images]

    #im = images[0]
    #plt.imshow(images[0])
    #plt.imsave("test.png", images[0])
    import matplotlib.pyplot as plt

    for i, img in enumerate(images):
        plt.imsave("test/test{}.png".format(i), img)

    create_video(images=images,
                 audio=wave_file.audio,
                 video_duration=wave_file.length,
                 sound_hz=wave_file.hz,
                 video_name="test.mp4")