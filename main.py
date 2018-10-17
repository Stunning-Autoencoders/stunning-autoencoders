from config.config import DEFAULT, CELEBA
from models.Vae import SimpleVAE, CelebAVAE
from utils.utils import create_video

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    #SimpleVAE(*DEFAULT).train()
    CelebAVAE(*CELEBA).train()
    # vae = SimpleVAE(*DEFAULT)
    # vae.load_pretrained("./training/SimpleVAE/run3")

    #create_video(SimpleVAE, DEFAULT, "./music/frequency_test.wav", "./training/SimpleVAE/run0",
    #             "frequency_test.mp4", fps=24)
