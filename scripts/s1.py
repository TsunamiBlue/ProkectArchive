import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

if __name__ == '__main__':
    model = torch.load(
        "D:\COMP-PROJECT-PROKECT\REPO\Dreambooth-Stable-Diffusion\models\ldm\stable-diffusion-v1\model.ckpt")
    for idx,key in enumerate(model["state_dict"].keys()):
        print(idx,key)
        if "model.diffusion_model.out" not in key:
            model["state_dict"][key].requires_grad=False
            print(model["state_dict"][key].requires_grad)


    print(len(model["state_dict"].keys()))



