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
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.integrate import solve_ivp

# Necessary to include this to use the files in the folder Functions
print("=" * 50)
print("Current path:" + os.path.dirname(os.path.abspath("")))
print("=" * 50)
sys.path.append(os.path.dirname(os.path.abspath("")))

import modules
import Functions.loss_functions as loss_functions
from Functions.training_modules import train_implicit_NeuralODE_Irregular
from Functions.models import linear_2D

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',
          # 'axes.labelpad': -10.0,
          }
plt.rcParams.update(params)




# For reproducbility
seed = 1234
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("No GPU found!")
else:
    print("Great, a GPU is there")
    # get_ipython().system('nvidia-smi')
print("=" * 50)

# Options
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

parser.add_argument("--timefinal", type=float, default=20.0)
parser.add_argument("--timestep", type=float, default=2e-1)
parser.add_argument("--dim_x", type=float, default=2)


parser.add_argument("--print_models", action="store_true")
parser.add_argument("--RK_back", action="store_true")

parser.add_argument("--noise_level", type=float, default=20e-2)
parser.add_argument("--logging_root", type=str, default="./logs")
parser.add_argument("--hidden_features_ODE", type=int, default=20)
parser.add_argument("--num_resblks_ODE", type=int, default=2)
parser.add_argument("--hidden_features", type=int, default=20)
parser.add_argument("--num_hidden_layers", type=int, default=3)
parser.add_argument("--lr_sol", type=float, default=5e-4)
parser.add_argument("--lr_grad", type=float, default=1e-3)

parser.add_argument("--params_datafidelity", type=float, default=1.0)
parser.add_argument("--params_neuralODE", type=float, default=1.0)
parser.add_argument("--params_grad", type=float, default=1e-2)
    
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--steps_til_summary", type=int, default=500)
parser.add_argument("--adjoint", action="store_true")

parser.add_argument("--data_size", type=int, default=50)
parser.add_argument("--batch_time", type=int, default=2)
#     parser.add_argument("--batch_size", type=int, default=2499)

opt = parser.parse_args()
    
experiment_name: str = (
    "./Linear2D_irregular/Implicit_NeuralODE/" + f"batchtime_{opt.batch_time}/"
)
opt.experiment_name = experiment_name

root_path = os.path.join(opt.logging_root, opt.experiment_name)
if not os.path.exists(root_path):
    os.makedirs(root_path)
    
print("Root path" + root_path)

######################################
########## DEFINE DATALOADER CLASS ###
######################################
class data_irregular(Dataset):
    def __init__(self , time , var):
        self.t = time
        self.p = var
        
        self.coords = torch.tensor(self.t).reshape(-1,1).float()
        self.func = torch.tensor(self.p).reshape(len(self.t), -1).float()
        
    def __len__(self):
        return 1
    
    def __getitem__(self , idx):
        return {"coords": self.coords}, {"func": self.func}
    
class data_onlytime(Dataset):
    def __init__(self , time ):
        self.t = time
        self.coords = self.t.reshape(-1,1).float
        
    def __len__(self):
        return 1
    
    def __getitem__(self , idx):
        return {"coords": self.t}
    

####################################
########## PLOTTING ################
####################################
def plotting_linear2D():
    # plot data
    fig = plt.figure(figsize=(4.5, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, x[0,:,0],'k--', linewidth=2)
    ax.plot(ts, x[0,:,1],'c--', linewidth=2)
    ax.plot(t_max*(data1['time']+1)/2, data1['value1'][0],'go', label = '$x(t)$', markersize=4)
    ax.plot(t_max*(data2['time']+1)/2, data2['value2'][0],'md', label = '$y(t)$', markersize=4)
    ax.set(xlabel = 'Time', ylabel = '{x(t),y(t)}')
    plt.legend()
    
    plt.tight_layout()

    
    fig.savefig(
        root_path + "measured_noisy_data_{}.pdf".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.savefig(
        root_path + "measured_noisy_data_{}.png".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
         dpi=150
    )
        
    # Plotting vector field on [-2,2]x[-2,2]
    v, w = np.meshgrid(np.linspace(-2.0, 2.0, 25), np.linspace(-2.0, 2, 25))
    
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
    
    fig = plt.figure(figsize=(12, 3))
    
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
        density=4,
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
        density=4,
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
        root_path + "Linear2D_vectorfield_noiselevel_{}.pdf".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
    )
    fig.savefig(
        root_path + "Linear2D_vectorfield_noiselevel_{}.png".format(opt.noise_level),
        bbox_inches="tight",
        pad_inches=0,
         dpi=150
    )


########################################################
################ Main script ##########################
########################################################
# Genrate data
dynModel = linear_2D

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
    
# Do a random sampling of measurement data
t_max = ts.max()
ts_nor = 2*(ts/t_max - 0.5)

idx = np.random.permutation(np.arange(len(ts_nor)))
len1 = int(len(ts_nor)*0.6)
len2 = len(ts_nor)-len1

x_noise = x + opt.noise_level * torch.randn_like(torch.from_numpy(x)).cpu().numpy()
data1 = {'time':ts_nor[idx[:len1]], 'value1': x_noise[:,idx[:len1],0]}
data2 = {'time':ts_nor[idx[len2:]], 'value2': x_noise[:,idx[len2:],1]}


# Crete dataloaders
dataset1 = data_irregular(data1['time'], data1['value1'])
dataloader1 = DataLoader(dataset1, shuffle=True, batch_size=1, num_workers=0)

dataset2 = data_irregular(data2['time'], data2['value2'])
dataloader2 = DataLoader(dataset2, shuffle=True, batch_size=1, num_workers=0)

dataset_time = data_onlytime(torch.tensor(ts_nor).to(device).float().unsqueeze(dim=-1))
dataloader_time = DataLoader(dataset_time, shuffle=True, batch_size=1, num_workers=0)

grid_info = {
    "tmax": t_max,
    "tlen": len(ts),
    "dt": ts[1] - ts[0],
    "prediction_ahead": opt.batch_time,
    "time_grid": ts
}
# print(grid_info['time_grid'])
print('-'*100)
print("Time span:          {}".format(grid_info["tmax"]))
print("Temporal points:    {}".format(grid_info["tlen"]))
print("Time-stepping (dt): {:.4f}".format(grid_info["dt"]))
print("Noise level:        {:.4f}".format(opt.noise_level))

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
}

# Define the loss
loss_fns = {
    "sol_predModel": loss_functions.function_mse,    "grad_predModel": loss_functions.simple_mse,
}

# Define optimizer
optim = torch.optim.Adam(
    [
        {"params": models["sol_predModel"].parameters()},
       {
            "params": models["neuralODE"].parameters(),
            "lr": opt.lr_grad,
            "weight_decay": 0.0,
        },
    ],
    lr=opt.lr_sol,
    weight_decay=1e-4,
)

# Training models
train_implicit_NeuralODE_Irregular(
    models=models,
    dataloader1 = dataloader1,
    dataloader2 = dataloader2,
    dataloader_time = dataloader_time,
    epochs=opt.num_epochs,
    steps_til_summary=opt.steps_til_summary,
    model_dir=root_path,
    loss_fn=loss_fns,
    optim=optim,
    grid_info=grid_info,
    decay_scheduler=True,
    params_datafidelity = opt.params_datafidelity,
    params_neuralODE = opt.params_neuralODE,
    params_grad = opt.params_grad,
)

# Plotting data
plotting_linear2D()

