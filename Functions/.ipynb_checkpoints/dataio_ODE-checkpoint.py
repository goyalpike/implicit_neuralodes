import csv
import glob
import math
import os
import scipy.io as sio

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def normalize_max(x):
    x_max = np.max(np.abs(x))
    return x / x_max, x_max


class Data_ODE(Dataset):
    def __init__(self, time, u, noise_level=0.0, filtering=False):
        super().__init__()
        self.noise_level = noise_level

        # Extract data
        self.u = u
        self.t = time
        print(self.t.shape)

        # Normalize data
        self.t_max = time.max()
        self.tnor = self.t / self.t_max

        self.tnor = 2 * (self.tnor - 0.5)

        # Data
        self.coords = torch.tensor(self.tnor.reshape(-1, 1)).float()
        self.u_vec = torch.tensor(self.u).reshape(len(self.tnor), -1).float()
        self.noise = torch.randn(*self.u_vec.shape) * noise_level

        self.u_vec += self.noise

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"coords": self.coords}, {"func": self.u_vec}
