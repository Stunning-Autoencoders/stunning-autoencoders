import imageio
import numpy as np
import cv2

imageio.plugins.ffmpeg.download()
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from typing import List

from music.music import WaveFile

DataList = List[np.ndarray]


def create_video(images: DataList, audio: np.ndarray, video_duration: float, sound_hz: int, video_name: str):
    """
    Creates video with sound from image list and music.

    Args:
        images: List of images represented by a h x w x 3 numpy array.
        audio: A Numpy array representing the sound, of size Nx1 for mono, Nx2 for stereo.
        video_duration: Duration of the video in seconds (should be the same as the audio file).
        sound_hz: The hz of the audio file.
        video_name: The name of the resulting video file
    """
    clip = ImageSequenceClip(images, durations=[video_duration / len(images)] * len(images))
    s = audio.reshape((len(audio), 1))  # transform it from (N) to (N, 1)
    audio = AudioArrayClip(s, sound_hz * 2)
    clip = clip.set_audio(audio)
    clip.write_videofile(video_name, fps=len(images) / video_duration)


if __name__ == '__main__':
    # load 100 previously stored images
    # and scale it to 255 so its visible
    images = np.load("../images.npy") * 255
    # now copy the first layer of the 3rd axis 2 times for fake rgb, its still black and white
    # at the same time resize the image to 320 x 320
    images = [cv2.resize(np.repeat(i, 3, axis=2), dsize=(320, 320)) for i in images]
    # load the music file
    wave_file = WaveFile(dir="../music/heartbeat.wav")
    # create the video
    # todo fix the audio
    create_video(images, wave_file.c1, 9, wave_file.hz, "first2.mp4")
