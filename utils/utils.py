import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from typing import List

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
    clip = ImageSequenceClip(images, durations=video_duration / len(images))
    clip = clip.set_audio(AudioArrayClip(audio, sound_hz))
    clip.write_videofile(video_name)
