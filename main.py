from config.config import DEFAULT, CELEBA
from models.Vae import SimpleVAE, CelebAVAE
#from utils.utils import create_video

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    #SimpleVAE(*DEFAULT).train()
    CelebAVAE(*CELEBA).train()
    #vae = SimpleVAE(*DEFAULT)
    #vae.load_pretrained("./training/CelebVAE/best_run")

    #create_video(CelebAVAE, CELEBA, "./music/we_will_rock_you.wav", "./training/CelebAVAE/next_best_run",
    #             "we_will_rock_you_test.mp4", fps=24)
