#!/usr/bin/env python3
# coding: utf-8

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

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})


path_implicit = "./Results/Cubic2D/Implicit_NeuralODE/"
path_neuralODE = "./Results/Cubic2D/NeuralODE/"

batchtimes = [2, 4, 6, 8, 10]
noise_level = np.array([0.0, 1e-2, 5e-2, 10e-2, 20e-2, 30e-2, 40e-2, 50e-2]).reshape(
    -1,
)

for batchtime in batchtimes:

    data_2_0 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.0.mat"
    )
    data_2_1 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.01.mat"
    )
    data_2_5 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.05.mat"
    )
    data_2_10 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.1.mat"
    )
    data_2_20 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.2.mat"
    )
    data_2_30 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.3.mat"
    )
    data_2_40 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.4.mat"
    )
    data_2_50 = scipy.io.loadmat(
        path_implicit + f"batchtime_{batchtime}/" + "error_0.5.mat"
    )

    data_2_0_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.0.mat"
    )
    data_2_1_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.01.mat"
    )
    data_2_5_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.05.mat"
    )
    data_2_10_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.1.mat"
    )
    data_2_20_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.2.mat"
    )
    data_2_30_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.3.mat"
    )
    data_2_40_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.4.mat"
    )
    data_2_50_NODE = scipy.io.loadmat(
        path_neuralODE + f"batchtime_{batchtime}/" + "error_0.5.mat"
    )

    data_2_mean_implicit = np.array(
        [
            data_2_0["mean_err"],
            data_2_1["mean_err"],
            data_2_5["mean_err"],
            data_2_10["mean_err"],
            data_2_20["mean_err"],
            data_2_30["mean_err"],
            data_2_40["mean_err"],
            data_2_50["mean_err"],
        ]
    ).reshape(
        -1,
    )
    data_2_median_implicit = np.array(
        [
            data_2_0["median_err"],
            data_2_1["median_err"],
            data_2_5["median_err"],
            data_2_10["median_err"],
            data_2_20["median_err"],
            data_2_30["median_err"],
            data_2_40["median_err"],
            data_2_50["median_err"],
        ]
    ).reshape(
        -1,
    )

    data_2_mean_NODE = np.array(
        [
            data_2_0_NODE["mean_err"],
            data_2_1_NODE["mean_err"],
            data_2_5_NODE["mean_err"],
            data_2_10_NODE["mean_err"],
            data_2_20_NODE["mean_err"],
            data_2_30_NODE["mean_err"],
            data_2_40_NODE["mean_err"],
            data_2_50_NODE["mean_err"],
        ]
    ).reshape(
        -1,
    )
    data_2_median_NODE = np.array(
        [
            data_2_0_NODE["median_err"],
            data_2_1_NODE["median_err"],
            data_2_5_NODE["median_err"],
            data_2_10_NODE["median_err"],
            data_2_20_NODE["median_err"],
            data_2_30_NODE["median_err"],
            data_2_40_NODE["median_err"],
            data_2_50_NODE["median_err"],
        ]
    ).reshape(
        -1,
    )

    fig = plt.figure(figsize=(6, 4))

    ax1 = fig.add_subplot(111)
    ax1.plot(noise_level, data_2_mean_implicit, "g*-", label="Implicit_NODE_mean")
    ax1.plot(noise_level, data_2_median_implicit, "g+-", label="Implicit_NODE_median")
    ax1.plot(noise_level, data_2_mean_NODE, "b*--", label="NeuralODE_mean")
    ax1.plot(noise_level, data_2_median_NODE, "b+--", label="NeuralODE_median")
    plt.legend()

    ax1.set(xlabel="Noise level", ylabel="error", title=f"batch_time_{batchtime}")

    fig.savefig(
        "./Results/Cubic2D/error_plot_batchsize{}.pdf".format(batchtime),
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.savefig(
        "./Results/Cubic2D/error_plot_batchsize{}.png".format(batchtime),
        bbox_inches="tight",
        pad_inches=0,
        dpi=150,
    )


# In[4]:


bt = np.array([2, 4, 6, 8, 10])
noise_level = 0.05
data_2_1 = scipy.io.loadmat(
    path_implicit + f"batchtime_2/" + f"error_{noise_level}.mat"
)
data_4_1 = scipy.io.loadmat(
    path_implicit + f"batchtime_4/" + f"error_{noise_level}.mat"
)
data_6_1 = scipy.io.loadmat(
    path_implicit + f"batchtime_6/" + f"error_{noise_level}.mat"
)
data_8_1 = scipy.io.loadmat(
    path_implicit + f"batchtime_8/" + f"error_{noise_level}.mat"
)
data_10_1 = scipy.io.loadmat(
    path_implicit + f"batchtime_10/" + f"error_{noise_level}.mat"
)

data_2_1_NODE = scipy.io.loadmat(
    path_neuralODE + f"batchtime_2/" + f"error_{noise_level}.mat"
)
data_4_1_NODE = scipy.io.loadmat(
    path_neuralODE + f"batchtime_4/" + f"error_{noise_level}.mat"
)
data_6_1_NODE = scipy.io.loadmat(
    path_neuralODE + f"batchtime_6/" + f"error_{noise_level}.mat"
)
data_8_1_NODE = scipy.io.loadmat(
    path_neuralODE + f"batchtime_8/" + f"error_{noise_level}.mat"
)
data_10_1_NODE = scipy.io.loadmat(
    path_neuralODE + f"batchtime_10/" + f"error_{noise_level}.mat"
)


data_mean_implicit = np.array(
    [
        data_2_1["mean_err"],
        data_4_1["mean_err"],
        data_6_1["mean_err"],
        data_8_1["mean_err"],
        data_10_1["mean_err"],
    ]
).reshape(
    -1,
)

data_median_implicit = np.array(
    [
        data_2_1["median_err"],
        data_4_1["median_err"],
        data_6_1["median_err"],
        data_8_1["median_err"],
        data_10_1["median_err"],
    ]
).reshape(
    -1,
)

data_mean_neuralODE = np.array(
    [
        data_2_1_NODE["mean_err"],
        data_4_1_NODE["mean_err"],
        data_6_1_NODE["mean_err"],
        data_8_1_NODE["mean_err"],
        data_10_1_NODE["mean_err"],
    ]
).reshape(
    -1,
)

data_median_neuralODE = np.array(
    [
        data_2_1_NODE["median_err"],
        data_4_1_NODE["median_err"],
        data_6_1_NODE["median_err"],
        data_8_1_NODE["median_err"],
        data_10_1_NODE["median_err"],
    ]
).reshape(
    -1,
)


fig = plt.figure(figsize=(6, 4))

ax1 = fig.add_subplot(111)
ax1.plot(bt, data_mean_implicit, "g*-", label="Implicit_neuralODE_mean")
ax1.plot(bt, data_median_implicit, "g+-", label="Implicit_neuralODE_median")
ax1.plot(bt, data_mean_neuralODE, "b*--", label="NeuralODE_mean")
ax1.plot(bt, data_median_neuralODE, "b+--", label="NeuralODE_median")
plt.grid(color="0.95")
plt.legend()

ax1.set(xlabel="Batch time", ylabel="error")

fig.savefig(
    "./Results/Cubic2D/error_plot_differentbatchsize.pdf",
    bbox_inches="tight",
    pad_inches=0,
)
fig.savefig(
    "./Results/Cubic2D/error_plot_differentbatchsize.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=150,
)
