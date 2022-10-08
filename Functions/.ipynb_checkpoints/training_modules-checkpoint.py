#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:34:18 2021

@author: goyalp
"""
import torch
import os, time
import Functions.utils as utils

from tqdm.autonotebook import tqdm
import Functions.diff_operators as diff_operators
from Functions.utils import get_batch_data
from torch.optim.lr_scheduler import StepLR
import numpy as np

from torchdiffeq import odeint
    
def train_implicitNN_ODE(
    models,
    train_dataloader,
    epochs,
    steps_til_summary,
    model_dir,
    loss_fn,
    optim,
    grid_info,
    decay_scheduler=None,
    epochs_decaylr=5000,
    RK_back=False,
    factor_datafidelity=1.0,
    params_rk=1.0,
    params_grad=1.0,
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tlen = grid_info["tlen"]
    tmax = grid_info["tmax"]
    dt = grid_info["dt"]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    if decay_scheduler:
        scheduler = StepLR(optim, step_size=epochs_decaylr, gamma=0.2)
    else:
        scheduler = StepLR(optim, step_size=epochs_decaylr, gamma=1.0)

    total_steps = 0
    total_time = 0.0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:

        train_losses = []
        train_losses_datamismatch = []
        train_losses_RK = []
        train_losses_grad = []

        for epoch in range(1, epochs + 1):

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                }
                gt = {key: value.to(device) for key, value in gt.items()}

                # Solution prediction directly from the network (e.g., SIREN)
                model_output = models["sol_predModel"](model_input)
                loss_solpred = loss_fn["sol_predModel"](model_output, gt)

                # First gradient ut and ux
                total_ut = diff_operators.gradient(
                    model_output["model_out"][..., 0], model_output["model_in"]
                )
                for i in range(1, model_output["model_out"].shape[-1]):
                    total_ut = torch.cat(
                        (
                            total_ut,
                            diff_operators.gradient(
                                model_output["model_out"][..., i],
                                model_output["model_in"],
                            ),
                        ),
                        dim=-1,
                    )

                total_ut = total_ut / (tmax / 2)

                # One step forward prediction by fusing the RK4 scheme
                u_pred = utils.rk4th_onestep(
                    models["grad_predModel"],
                    model_output["model_out"][:, :-1, :],
                    timestep=dt,
                )
                loss_RKforw = loss_fn["grad_predModel"](
                    u_pred, model_output["model_out"][:, 1:, :]
                )

                if RK_back:
                    # One step backward prediction by fusing the RK4 scheme
                    u_back = utils.rk4th_onestep(
                        models["grad_predModel"],
                        model_output["model_out"][:, 1:, :],
                        timestep=-dt,
                    )
                    loss_RKback = loss_fn["grad_predModel"](
                        u_back, model_output["model_out"][:, :-1, :]
                    )

                # Derivative from the network and RK
                ut_pred = models["grad_predModel"](
                    model_output["model_out"].reshape(1, tlen, -1)
                )
                loss_grad = loss_fn["grad_predModel"](ut_pred, total_ut)

                #################################
                ## Assembling losses ############
                #################################
                train_loss = 0.0
                # Prediction loss
                for loss_name, loss in loss_solpred.items():
                    single_loss = loss.mean()

                    train_loss += factor_datafidelity * single_loss
                    train_losses_datamismatch.append(single_loss.item())

                # RK loss
                for loss_name, loss in loss_RKforw.items():
                    single_loss = loss.mean()

                    train_loss += params_rk * single_loss
                    train_losses_RK.append(single_loss.item())

                if RK_back:
                    for loss_name, loss in loss_RKback.items():
                        single_loss = loss.mean()

                        train_loss += params_rk * single_loss

                #                 Derivative mismatch loss from RK and network models
                for loss_name, loss in loss_grad.items():
                    single_loss = params_grad * loss.mean()

                    train_loss += single_loss
                    train_losses_grad.append(single_loss.item())

                
                # 
                # grad_l1 = 1e-4*ut_pred.abs().mean()
                # train_loss += grad_l1

                train_losses.append(train_loss.item())

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                scheduler.step()
                pbar.update(1)
                total_time += time.time() - start_time

                if not epoch % steps_til_summary:
                    torch.save(
                        models["sol_predModel"].state_dict(),
                        os.path.join(
                            checkpoints_dir, "model_sol_predModel_current.pth"
                        ),
                    )
                    torch.save(
                        models["grad_predModel"].state_dict(),
                        os.path.join(
                            checkpoints_dir, "model_grad_predModel_current.pth"
                        ),
                    )

                if not epoch % steps_til_summary:
                    tqdm.write(
                        "Epoch %d, Total loss %0.6e, sol loss %0.6e, RK loss %0.6e grad loss %0.6e, iteration time %0.6f, factor %0.6f"
                        % (
                            epoch,
                            train_loss,
                            loss_solpred["func_loss"].item(),
                            loss_RKforw["func_loss"].item(),
                            loss_grad["func_loss"].item(),
                            total_time / (epoch),
                            factor_datafidelity,
                        )
                    )
                total_steps += 1
    return {
        "train_loss": train_losses,
        "train_data": train_losses_datamismatch,
        "train_RK": train_losses_RK,
        "train_grad": train_losses_grad,
    }, {"AvgTime": total_time / epochs}


def train_implicit_NeuralODE(
    models,
    train_dataloader,
    epochs,
    steps_til_summary,
    model_dir,
    loss_fn,
    optim,
    grid_info,
    decay_scheduler=None,
    epochs_decaylr=4000,
    RK_back=False,
    params_datafidelity = 1.0,
    params_neuralODE = 1.0,
    params_grad = 1.0,
):
    
    print(f"params_datafidelity: {params_datafidelity}   params_neuralODE: {params_neuralODE}   params_grad: {params_grad}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tlen = grid_info["tlen"]
    tmax = grid_info["tmax"]
    dt = grid_info["dt"]
    q = grid_info["prediction_ahead"]
    
    t = torch.tensor(grid_info["time_grid"]).float().to(device)
    
    print("prediction ahead {}".format(q))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    if decay_scheduler:
        scheduler = StepLR(optim, step_size=epochs_decaylr, gamma=0.1)
    else:
        scheduler = StepLR(optim, step_size=epochs_decaylr, gamma=1.0)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(1, epochs+1):
            
            for step, (model_input, gt) in enumerate(train_dataloader):
                time_iter = time.time()
                optim.zero_grad()
                
                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                }
                gt = {key: value.to(device) for key, value in gt.items()}

                # Solution prediction directly from the network (e.g., SIREN)
                model_output = models["sol_predModel"](model_input)
#                 print(model_output['model_out'].shape)
                            
                # First gradient ut and ux
                total_ut = diff_operators.gradient(
                    model_output["model_out"][..., 0], model_output["model_in"]
                )
                for i in range(1, model_output["model_out"].shape[-1]):
                    total_ut = torch.cat((total_ut,
                            diff_operators.gradient(
                                model_output["model_out"][..., i],
                                model_output["model_in"],
                            ),
                        ),
                        dim=-1,
                    )

                total_ut = total_ut / (tmax / 2)
                
                # Derivative from the network and RK
                ut_pred = models["neuralODE"](0,
                    model_output["model_out"].reshape(1, tlen, -1)
                )
                        
                y = model_output['model_out'][0].unsqueeze(dim=-2)
                
                batch_y0, batch_t, batch_y = get_batch_data(y, t, batch_time = q)
                batch_y0, batch_t, batch_y = batch_y0.to(device), batch_t.to(device), batch_y.to(device)
                
                if epoch == 1:
                    print(batch_y0.shape, batch_t.shape, batch_y.shape)

                pred_y = odeint(models['neuralODE'], batch_y0, batch_t).to(device)
                
                loss_solpred = loss_fn["sol_predModel"](model_output, gt)
                loss_grad = loss_fn["grad_predModel"](ut_pred, total_ut)
                
                # grad_l1 = 1e-4*ut_pred.abs().mean()


                
                loss_neuralODE = (pred_y - batch_y).pow(2).mean()
                
                loss = params_neuralODE*loss_neuralODE \
                    + params_datafidelity*loss_solpred["func_loss"] \
                        + params_grad*loss_grad["func_loss"]
                
                loss.backward()
                optim.step()
                scheduler.step()
                pbar.update(1)                


                if not epoch % steps_til_summary:
                    tqdm.write(
                        "Epoch {:04d} | Total Loss {:.4e} | Loss Implicit {:.4e} | Loss NeuralODE {:.4e} | Loss Grad {:.4e} | Time {:.4f}".format(
                                epoch, loss.item(), loss_solpred["func_loss"].item(), loss_neuralODE.item(), loss_grad["func_loss"], time.time() - time_iter
                            )
                    )
                total_steps += 1


def train_NeuralODE(
    models,
    train_dataloader,
    epochs,
    steps_til_summary,
    model_dir,
    loss_fn,
    optim,
    grid_info,
    decay_scheduler=None,
    epochs_decaylr=4000,
    RK_back=False,
    params_datafidelity = 1.0,
    params_neuralODE = 1.0,
    params_grad = 1.0,
):
    
    print(f"params_datafidelity: {params_datafidelity}   params_neuralODE: {params_neuralODE}   params_grad: {params_grad}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q = grid_info["prediction_ahead"]
    
    t = torch.tensor(grid_info["time_grid"]).float().to(device)
    
    print("prediction ahead {}".format(q))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    if decay_scheduler:
        scheduler = StepLR(optim, step_size=epochs_decaylr, gamma=0.1)
    else:
        scheduler = StepLR(optim, step_size=epochs_decaylr, gamma=1.0)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        for epoch in range(1, epochs+1):
            
            for step, (model_input, gt) in enumerate(train_dataloader):
                time_iter = time.time()
                optim.zero_grad()
                
                model_input = {
                    key: value.to(device) for key, value in model_input.items()
                }
                gt = {key: value.to(device) for key, value in gt.items()}
                        
                y = gt['func'][0].unsqueeze(dim=-2)
                
                batch_y0, batch_t, batch_y = get_batch_data(y, t, batch_time = q)
                batch_y0, batch_t, batch_y = batch_y0.to(device), batch_t.to(device), batch_y.to(device)
                
                if epoch == 1:
                    print(batch_y0.shape, batch_t.shape, batch_y.shape)

                pred_y = odeint(models['neuralODE'], batch_y0, batch_t).to(device)

                
                loss_neuralODE = torch.mean(torch.abs(pred_y - batch_y))
                
                loss = params_neuralODE*loss_neuralODE 
                
                loss.backward()
                optim.step()
                scheduler.step()
                pbar.update(1)                


                if not epoch % steps_til_summary:
                    tqdm.write(
                        "Epoch {:04d} | Total Loss {:.4e} | Loss NeuralODE {:.4e} | Time {:.4f}".format(
                                epoch, loss.item(), loss_neuralODE.item(),time.time() - time_iter
                            )
                    )
                total_steps += 1
