from config.config import DEFAULT
from models.Vae import SimpleVAE

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # SimpleVAE(*DEFAULT).train()
    vae = SimpleVAE(*DEFAULT)
    # vae.load_pretrained("./training/SimpleVAE/run3")
    vae.generate_images_from_audio(weights="./training/SimpleVAE/run3",
                                   file="music/heartbeat.wav",
                                   interval=2,
                                   samples=20)

