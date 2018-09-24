from config.config import DEFAULT
from models.Vae import SimpleVAE

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    vae = SimpleVAE(*DEFAULT)#.train()
    vae.load_pretrained("./training/SimpleVAE/2018-09-24 13:50:25.226493")
