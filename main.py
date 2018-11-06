from config.config import DEFAULT, CELEBA
from models.Vae import SimpleVAE, CelebAVAE
from utils.utils import create_video

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # SimpleVAE(*DEFAULT).train()
    CelebAVAE(*CELEBA).train()
    # file_name = ""
    # create_video(CelebAVAE, CELEBA, f"./music/{file_name}.wav", "./training/CelebAVAE/100_hidden",
    #             f"{file_name}_test.mp4", fps=24)
