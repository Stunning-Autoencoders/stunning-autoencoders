from config.config import DEFAULT
from models.Vae import SimpleVAE
from utils.utils import create_video

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    #SimpleVAE(*DEFAULT).train()
    #vae = SimpleVAE(*DEFAULT)
    #vae.load_pretrained("./training/SimpleVAE/run2")

    create_video(SimpleVAE, DEFAULT, "./music/heartbeat.wav", "./training/SimpleVAE/run3",
                 "heartbeat.mp4", fps=24)
