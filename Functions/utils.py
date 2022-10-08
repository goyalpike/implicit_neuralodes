
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
import os
import torch
import numpy as np

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    ## Simple RK model


def rk4th_onestep(model, x, t=0, timestep=1e-2, num_steps=1):

    h = timestep
    k1 = model(x)
    k2 = model(x + 0.5 * h * k1)
    k3 = model(x + 0.5 * h * k2)
    k4 = model(x + 1.0 * h * k3)
    y = x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * h

    return y


def get_batch_data(y, t, batch_time = 2):
    data_size = y.shape[0]
    
    s = torch.from_numpy(np.arange(data_size - batch_time + 1, dtype=np.int64))
    # s = np.random.permutation(s)
    # s = torch.from_numpy(s)[:100]
    
    batch_y0 = y[s]  # (M, D)
    batch_t = t[: batch_time]  # (T)
    batch_y = torch.stack(
        [y[s + i] for i in range(batch_time)], dim=0
    )  # (T, M, D)
    return batch_y0, batch_t, batch_y
