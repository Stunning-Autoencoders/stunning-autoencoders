import imageio
import numpy as np
import cv2

# make sure ffmpeg is installed
from config.config import DEFAULT
from models.Vae import SimpleVAE
from models.Vae import VAE

imageio.plugins.ffmpeg.download()
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from typing import List, Type

from music.music import WaveFile

DataList = List[np.ndarray]


def merge_images_and_audio(images: DataList, audio: np.ndarray, video_duration: float, sound_hz: int, video_name: str):
    """
    Creates video with sound from image list and music.

    Args:
        images: List of images represented by a h x w x 3 numpy array.
        audio: A Numpy array representing the sound, of size Nx1 for mono, Nx2 for stereo.
        video_duration: Duration of the video in seconds (should be the same as the audio file).
        sound_hz: The hz of the audio file.
        video_name: The name of the resulting video file
    """
    # todo there is still a problem with the audio here
    # the audio should always contain two channels
    # then the hz should also work for mono and dual
    clip = ImageSequenceClip(images, durations=[video_duration / len(images)] * len(images))
    s = audio.reshape((len(audio), 2))  # transform it from (N) to (N, 2)
    audio = AudioArrayClip(s, sound_hz)
    clip = clip.set_audio(audio)
    clip.write_videofile(video_name, fps=len(images) / video_duration)


def create_video(VAE: Type[VAE], config, audio_file: str, model_path: str, output_file, fps=100):
    vae = VAE(*config)

    wave_file = WaveFile(dir=audio_file)
    sampled_dists = wave_file.generate_mell(intervals=fps, samples=config.hidden_size)
    #sampled_dists = wave_file.generate_samples(intervals=fps, samples=20)

    im1 = vae.generate_images(weights=model_path,
                                 dists=sampled_dists[:len(sampled_dists)//2])

    im2 = vae.generate_images(weights=model_path,
                                 dists=sampled_dists[len(sampled_dists)//2:])

    images = np.concatenate((im1, im2))

    images = (images[:, :, :, :] * 255.999).astype(np.uint8)
    #images = np.stack((images, images, images), -1)

    #for i, image in enumerate(images):
    #    image = cv2.resize(image, dsize=(138, 138))
    #    plt.imsave("generated_images/heartbeat/image_{}.png".format(i), image)

    images = [cv2.resize(i, dsize=(320, 320)) for i in images]

    merge_images_and_audio(images=images,
                           audio=wave_file.audio,
                           video_duration=wave_file.length,
                           sound_hz=wave_file.hz,
                           video_name=output_file)


if __name__ == '__main__':
    # create_video(SimpleVAE, DEFAULT, "../music/heartbeat.wav", "../training/SimpleVAE/run0", "heartbeat.mp4")
    # create_video(SimpleVAE, DEFAULT, "../music/airplane-landing.wav", "../training/SimpleVAE/run0", "airplane-landing.mp4")
    create_video(SimpleVAE, DEFAULT, "../music/heartbeat.wav", "../training/SimpleVAE/run3",
                 "heartbeat.mp4", fps=24)
