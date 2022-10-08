
"""
MIT License

Copyright (c) 2022 Pawan Goyal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 
Author: Pawan Goyal
"""
# Imported required packages

import numpy as np
import torch
from torch.utils.data import Dataset


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
