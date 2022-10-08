#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

import os, sys
import argparse
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.integrate import solve_ivp

# Necessary to include this to use the files in the folder Functions
print("=" * 50)
print("Current path:" + os.path.dirname(os.path.abspath("")))
print("=" * 50)
sys.path.append(os.path.dirname(os.path.abspath("")))

import Functions.dataio_ODE as dataio_ODE
import modules
import Functions.loss_functions as loss_functions
from Functions.training_modules import  train_implicit_NeuralODE, train_implicit_RKODE

from Functions.models import cubic_2D
from scipy.io import savemat

params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',
          }
plt.rcParams.update(params)

# For reproducbility
seed = 3407
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


# Setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("No GPU found!")
else:
    print("Great, a GPU is there")
print("=" * 50)

"""
These are arugements one may provide using parser

method:                 it allows to choose numerical intergrarion method
timefinal:              end time to integrate the dynamical model
timestep:               time interval at which data are collected
dim_x:                  the dimension of the model or number of variables
print_models:           the printing of the networks for vector fields and implicit respresentation
noise_level:            noise level in the data
logging_root:           path to store all logs 
hidden_features_ODE:    dimensional of the hidden layer in the network defining vector field
num_resblks_ODE:        number of residual blocks in the network defining vector field
hidden_features:        dimensional of the hidden layer in the network for implicit respresentation
num_hidden_layers:      number of residual blocks in the network for implicit respresentation
lr_sol:                 learning rate for training the implicit respresentation network
lr_grad:                learning rate for training the network defining  vector field

params_datafidelity:    weightage of the data didelity loss
params_neuralODE:       weightage of the neural ODE loss
params_grad:            weightage of the mismatch of the gradients

num_epochs:             number of epochs for training
steps_til_summary:      potting/printing summary after that many epochs
batch_time:             defines the time-interval for integration in neural ODEs

"""

parser = argparse.ArgumentParser("")
parser.add_argument("--method", type=str, choices=["dopri5", "adams"], default="dopri5")
parser.add_argument("--niters", type=int, default=2000)

parser.add_argument("--timefinal", type=float, default=25.0)
parser.add_argument("--timestep", type=float, default=1e-2)
parser.add_argument("--dim_x", type=float, default=2)


parser.add_argument("--print_models", action="store_true")
parser.add_argument("--RK_back", action="store_true")

parser.add_argument("--noise_level", type=float, default=1e-2)
parser.add_argument("--logging_root", type=str, default="./logs")
parser.add_argument("--hidden_features_ODE", type=int, default=20)
parser.add_argument("--num_resblks_ODE", type=int, default=4)
parser.add_argument("--hidden_features", type=int, default=20)
parser.add_argument("--num_hidden_layers", type=int, default=4)
parser.add_argument("--lr_sol", type=float, default=1e-3)
parser.add_argument("--lr_grad", type=float, default=1e-3)
#
parser.add_argument("--params_datafidelity", type=float, default=1.0)
parser.add_argument("--params_neuralODE", type=float, default=1.0)
parser.add_argument("--params_grad", type=float, default=1e-2)

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--steps_til_summary", type=int, default=500)
parser.add_argument("--adjoint", action="store_true")

parser.add_argument("--data_size", type=int, default=2500)
parser.add_argument("--batch_time", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=2499)


opt = parser.parse_args()

   
experiment_name: str = (
    "./Cubic2D/Implicit_NeuralODE/" + f"batchtime_{opt.batch_time}/"
)
opt.experiment_name = experiment_name

# Setting root path
root_path = os.path.join(opt.logging_root, opt.experiment_name)

print("Root path" + root_path)

if not os.path.exists(root_path):
    os.makedirs(root_path)

# Genrate data
dynModel = cubic_2D

def plotting_Cubic2D():
    # Plotting vector field on [-2,2]x[-2,2]
    # Meshgrid
    v, w = np.meshgrid(np.linspace(-2.0, 2.0, 25), np.linspace(-2.0, 2.0, 25))
    
    dv_truth, dw_truth = np.zeros_like(v), np.zeros_like(w)
    dv_model, dw_model = np.zeros_like(v), np.zeros_like(w)
    
    # # Directional vectors
    for i in range(25):
        for j in range(25):
            v1, w1 = v[i, j], w[i, j]
            vw = np.array([v1, w1])
            dz = dynModel(vw, 0)
            dz1 = (
                models["neuralODE"](0,torch.from_numpy(vw).float().to(device))
                .detach()
                .cpu()
                .numpy()
            )
    
            dv_truth[i, j], dw_truth[i, j] = dz[0], dz[1]
            dv_model[i, j], dw_model[i, j] = dz1[0], dz1[1]
    
    err = np.sqrt((dv_truth - dv_model) ** 2 + (dw_truth - dw_model) ** 2)
    
    fig = plt.figure(figsize=(13, 3.5))
    
    ax1 = fig.add_subplot(131)
    
    color = np.sqrt(dv_truth ** 2 + dw_truth ** 2)
    h1 = ax1.streamplot(
        v,
        w,
        dv_truth,
        dw_truth,
        color=color,
        linewidth=1,
        cmap=plt.cm.inferno,
        density=2,
        arrowstyle="->",
        arrowsize=1.5,
    )
    cbar = fig.colorbar(h1.lines, ax=ax1)
    ax1.set(title="", ylabel="$y$", xlabel="$x$")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    
    ax2 = fig.add_subplot(132)
    color = np.sqrt(dv_model ** 2 + dw_model ** 2)
    
    h2 = ax2.streamplot(
        v,
        w,
        dv_model,
        dw_model,
        color=color,
        linewidth=1,
        cmap=plt.cm.inferno,
        density=2,
        arrowstyle="->",
        arrowsize=1.5,
    )
    cbar = fig.colorbar(h2.lines, ax=ax2)
    
    ax2.set(title="", ylabel="$y$", xlabel="$x$")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)
    
    ax3 = fig.add_subplot(133)
    plt.contourf(v, w, err, 100, cmap=plt.cm.inferno)
    plt.colorbar(format='%.2f')
    ax3.set(title="", ylabel="$y$", xlabel="$x$")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    plt.tight_layout()

    fig.savefig(
        root_path + "Cubic2D_vectorfield_noiselevel_{}.pdf".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.savefig(
        root_path + "Cubic2D_vectorfield_noiselevel_{}.png".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
    )
    
    plt.draw()
    plt.pause(0.1)
    
    
    ############
    # Plotting denoised signal
    for step, (model_input, gt) in enumerate(dataloader):
    
        model_input = {key: value.to(device) for key, value in model_input.items()}
        gt = {key: value.to(device) for key, value in gt.items()}
    
        # Solution prediction directly from the network (e.g., SIREN)
        model_output = models["sol_predModel"](model_input)
    
        x_noise = gt["func"].detach().cpu().numpy()
        x_denoised_implicit = model_output["model_out"].detach().cpu().numpy()
    
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(
        ts, x_noise[0, ..., 0], "g", linewidth=3, label="Noisy measurements", markersize=3
    )
    ax.plot(ts, x_noise[0, ..., 1], "g", linewidth=3, markersize=3)
    
    ax.plot(ts, x[0, :, 0], "k", linewidth=3.5, label="Clean data")
    ax.plot(ts, x[0, :, 1], "k", linewidth=3.5)
    
    ax.plot(ts, x_denoised_implicit[0, ..., 0], "m--", linewidth=3, label="Denoised data")
    ax.plot(ts, x_denoised_implicit[0, ..., 1], "m--", linewidth=3)
    ax.set(ylabel="$\{v(t),w(t)\}$", xlabel="Time ($t$)")
    ax.legend()
    
    plt.tight_layout()
    
    fig.savefig(
        root_path + "Cubic2D_Signal_noiselevel_{}.pdf".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.savefig(
        root_path + "Cubic2D_Signal_noiselevel_{}.png".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
    )
    
    plt.draw()
    plt.pause(0.1)
    
    
    if opt.noise_level > 0.0:
        fig = plt.figure(figsize=(24, 4))
        ax = fig.add_subplot(1, 4, 1)
        ax.plot(ts, true_x[0,...,0], linewidth = 3, label = 'x(t)')
        ax.plot(ts, true_x[0,...,1], linewidth = 3, label = 'y(t)')

        ax.set(title="", ylabel="$\{x,y\}$", xlabel="Time ($t$)")
        ax.legend(ncol=2)
    
        ax = fig.add_subplot(1, 4, 2)
        ax.plot(ts, noise_x[0,...,0], linewidth = 3, label = 'x(t)')
        ax.plot(ts, noise_x[0,...,1], linewidth = 3, label = 'y(t)')
        ax.set(title="", ylabel="$\{x,y\}$", xlabel="Time ($t$)")
        ax.legend(ncol=2)

    
        ax = fig.add_subplot(1, 4, 3)
        ax.plot(ts, denoise_x[0,...,0], linewidth = 3,  label = 'x(t)')
        # ax.plot(ts, true_x[0,...,0], "--", linewidth = 3)
        ax.plot(ts, denoise_x[0,...,1], linewidth = 3,  label = 'y(t)')
        # ax.plot(ts, true_x[0,...,1], "--", linewidth = 3)
        ax.legend(ncol=2)

        ax.set(
            title="", ylabel="$\{x,y\}$", xlabel="Time ($t$)"
        )
    
        ax = fig.add_subplot(1, 4, 4)
        ax.plot(ts, x_denoised_implicit[0,...,0], linewidth = 3, label = 'x(t)')
        # ax.plot(ts, true_x[0,...,0], 'ko', markevery=20, markersize=2)
        ax.plot(ts, x_denoised_implicit[0,...,1], linewidth = 3, label = 'y(t)')
        # ax.plot(ts, true_x[0,...,1], 'ko',markevery=20, markersize=2)
        ax.legend(ncol=2)

        ax.set(
            title="", ylabel="$\{x,y\}$", xlabel="Time ($t$)"
        )
        plt.legend()
        plt.tight_layout()

    
        fig.savefig(
            root_path + "denoising_filtering_{}.pdf".format(opt.noise_level),
            bbox_inches="tight",
            pad_inches=0,
        )
        fig.savefig(
            root_path + "denoising_filtering_{}.png".format(opt.noise_level),
            bbox_inches="tight",
            pad_inches=0,
        )

    
    return err


######################################################
################ Main Script #########################
######################################################
print("="*50)
print(f"=======  Starting with noise {opt.noise_level} and batch time {opt.batch_time}")
print("="*50)
      
ts = np.arange(0, opt.timefinal, opt.timestep)
# Initial condition and simulation time
x0 = [2, 0]
# Solve the equation
sol = solve_ivp(lambda t, x: dynModel(x, t), [ts[0], ts[-1]], x0, t_eval=ts)
# x = sol
x = np.transpose(sol.y).reshape(1, -1, opt.dim_x)

# True vector field at these points
dx = np.zeros_like(x)
for i in range(len(ts)):
    dx[0, i, :] = dynModel(x[0, i, :], 0)
    

if opt.noise_level > 0.0:
    true_x = x
    noise_x = (
        true_x
        + opt.noise_level * torch.randn_like(torch.from_numpy(true_x)).cpu().numpy()
    )

    from scipy import signal

    b, a = signal.butter(3, 0.1, btype="lowpass", analog=False)
    denoise_x = np.zeros_like(true_x)

    denoise_x[0, ..., 0] = signal.filtfilt(b, a, noise_x[0, ..., 0])
    denoise_x[0, ..., 1] = signal.filtfilt(b, a, noise_x[0, ..., 1])
    

# Define dataloaders
if opt.noise_level == 0.0:
    dataset = dataio_ODE.Data_ODE(ts, x, noise_level=0.0)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
else:
    dataset = dataio_ODE.Data_ODE(ts, denoise_x, noise_level=0.0)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    
    
    
grid_info = {
    "tmax": dataset.t_max,
    "tlen": len(dataset.t),
    "dt": dataset.t[1] - dataset.t[0],
    "prediction_ahead": opt.batch_time,
    "time_grid": ts
}

print("Time span:          {}".format(grid_info["tmax"]))
print("Temporal points:    {}".format(grid_info["tlen"]))
print("Time-stepping (dt): {:.4f}".format(grid_info["dt"]))
print("Noise level:        {:.4f}".format(opt.noise_level))
print("Prediction Ahead:        {:.4f}".format(opt.batch_time -1))


# Define models
models = {
    "sol_predModel": modules.SingleBVPNet(
        in_features=1,
        out_features=2,
        type="sine",
        mode="mlp",
        final_layer_factor=1.0,
        hidden_features=opt.hidden_features,
        num_hidden_layers=opt.num_hidden_layers,
        print_model=opt.print_models,
    ).to(device),
    "neuralODE": modules.ODE_Net_NeuralODE(
        n=2,
        num_residual_blocks=opt.num_resblks_ODE,
        hidden_features=opt.hidden_features_ODE,
        print_model=opt.print_models,
    ).to(device),
    "grad_predModel": modules.ODE_Net(
        n=2,
        num_residual_blocks=opt.num_resblks_ODE,
        hidden_features=opt.hidden_features_ODE,
        print_model=opt.print_models,
    ).to(device),
}

# Define the loss
loss_fns = {
    "sol_predModel": loss_functions.function_mse, "grad_predModel": loss_functions.simple_mse,
}

# Define optimizer
optim = torch.optim.Adam(
    [
        {"params": models["sol_predModel"].parameters()},
        {
            "params": models["grad_predModel"].parameters(),
            "lr": opt.lr_grad,
            "weight_decay": 1e-4,
        },
    ],
    lr=opt.lr_sol,
    weight_decay=1e-4,
)

# Pre-training with RK scheme since it takes only four fix steps, it is faster though it might be 
# memory less-efficient as comapred to NeuralODE

losses, avgtime = train_implicit_RKODE(
    models=models,
    train_dataloader=dataloader,
    epochs=5000,
    steps_til_summary=opt.steps_til_summary,
    model_dir=root_path,
    loss_fn=loss_fns,
    optim=optim,
    grid_info=grid_info,
    decay_scheduler=True,
    RK_back=opt.RK_back,
    factor_datafidelity=50*opt.params_datafidelity,
    params_rk = 1.0,
    params_grad = 1.0,
)

# Copy the model parameters
models['neuralODE'].load_state_dict(models['grad_predModel'].state_dict())

# Define optimizer
optim = torch.optim.Adam(
    [
        {"params": models["sol_predModel"].parameters()},
       {
            "params": models["neuralODE"].parameters(),
            "lr": opt.lr_grad/5,
            "weight_decay": 0e-4,
        },
    ],
    lr=opt.lr_sol/5,
    weight_decay=1e-4,
)
    
# Training models
train_implicit_NeuralODE(
    models=models,
    train_dataloader=dataloader,
    epochs=opt.num_epochs,
    steps_til_summary=opt.steps_til_summary,
    model_dir=root_path,
    loss_fn=loss_fns,
    optim=optim,
    grid_info=grid_info,
    decay_scheduler=True,
    params_datafidelity=opt.params_datafidelity,
    params_grad = opt.params_grad,
    params_neuralODE = opt.params_neuralODE
)
    
# Plotting figures and obtain error
err = plotting_Cubic2D()

mean_err = np.mean(err)
median_err = np.median(err)

data_err = {"mean_err": np.mean(err), "median_err":  np.median(err)}
savemat(root_path + "error_{}.mat".format(opt.noise_level), data_err)


print(f"Error mean   (Noise-{opt.noise_level}):   {np.mean(err)}")
print(f"Error median (Noise-{opt.noise_level}):   {np.median(err)}")
print("*"*100)
print("\n\n\n")
