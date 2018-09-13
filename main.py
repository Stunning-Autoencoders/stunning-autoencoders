from config.config import DEFAULT
from models.Vae import SimpleVAE

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    SimpleVAE(*DEFAULT).train()
