import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.preprocessing import calculate_projection
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.gaussian_smooth import GaussianSmoothing
from utils.image_utils import image_gradient_3d
from reconstruction_models.layers import projection

from reconstruction_models.reconstruct_model import (nccSim, plot_rec_result,
                               print_sim_of_all, sim)

parser = argparse.ArgumentParser(description='3D/2D reconstruction')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')
parser.add_argument('--data', default='dirlab')
parser.add_argument('--result','-r', default='./result')
parser.add_argument('--preprocess','-p',default="")
parser.add_argument('--exp','-e',default="exp")
parser.add_argument('--with_prior',type=int,default=0)
parser.add_argument('--with_orth_proj',type=int,default=0)
parser.add_argument('--angle', type=int, default=0,
                    help='The scanning range of angles.') 
parser.add_argument('--projection_num', type=int, default=0,
                    help='The number of projection used.')   
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def reconstruct(I_proj, I_target, poses, spacing, I_init, mu, lam, eps,
                I_source = None, 
                epochs = 400, log_step=100, I_seg=None, I_warped=None, I_proj_orth=None, 
                log_path = "./log", result_path="./result", pose_scale="", lr_init=1e-04
                ):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    resolution = [I_proj.shape[2], I_proj.shape[3]]
    sample_rate = [int(1), int(1), int(1)]
    I_target_npy = I_target.cpu().numpy()[0,0]
    model = projection(I_target.shape, resolution, sample_rate, spacing, I_init, None)
    # opt = torch.optim.Adam(model.parameters(), lr=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr_init)
    # opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    # opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5, cooldown=10, verbose=True)

    gaussian_smooth = GaussianSmoothing(1, 10, 5, dim=3).to(device)
    pooling = torch.nn.MaxPool3d(5, 3)
    if I_seg is not None:
        I_seg = 1-I_seg

    loss_log = []
    for i in range(epochs):
        opt.zero_grad()
        output = model(poses)
        # output = model()
        for p in model.parameters():
            if I_warped is not None:
                sim_loss_3d = sim(pooling(gaussian_smooth(I_warped)), pooling(gaussian_smooth(p)), 'mean')
            else: 
                sim_loss_3d = torch.sum(F.relu(-1*p)) + torch.mean(image_gradient_3d(p, device))
            total_loss = sim_loss_3d

            # orth_loss = sim(torch.sum(p,dim=4), I_proj_orth, reduction="mean")
            orth_loss = torch.tensor(0).to(device)
            # relu_loss = torch.sum((p*I_seg)**2)#torch.sum(F.relu(-1*p))
            relu_loss = torch.tensor(0).to(device)
        
        loss = torch.tensor(0).to(device)
        loss_before_lam = torch.tensor(0).to(device)
        for j in range(4):
            loss_before_lam = loss_before_lam + sim(output[j], I_proj[j], 'mean')
            loss = loss + lam[j]*sim(output[j], I_proj[j], 'mean')
        
        total_loss = total_loss - loss - lam[4]*orth_loss - lam[5]*relu_loss \
            + mu*((torch.sum(loss_before_lam)+orth_loss+relu_loss)**2)/2
        
        # record the loss
        loss_log.append([total_loss.item(), loss_before_lam.item(), sim_loss_3d.item(),
                         orth_loss.item(), relu_loss.item()])

        # Calculate the derivative
        total_loss.backward()

        # Should we terminate?
        should_term = False
        for p in model.parameters():
            gradient = torch.sum(p.grad**2)
            if gradient < eps**2 or gradient == eps**2:
                should_term = True
                break
        if should_term:
            break

        opt.step()
        scheduler.step(total_loss)


        if i%log_step == 0:
            loss_at_it = loss_log[i]
            proj_ncc_loss = nccSim(output, I_proj)
            print("Epoch %i: total loss: %10.3e, rec loss: %10.3e, 3D sim loss: %10.3e, orth loss: %10.3e, relu loss: %10.3e | gradient: %10.3e"%(i, loss_at_it[0], loss_at_it[1], loss_at_it[2], loss_at_it[3], loss_at_it[4], gradient))

    # At the end of optimization, plot output
    for p in model.parameters():
        # diff = l1_loss(p, I_target, reduction='mean')
        loss_at_it = loss_log[i]
        proj_ncc_loss = nccSim(output, I_proj)
        print("End epoch %i: total loss: %10.3e, rec loss: %10.3e, 3D sim loss: %10.3e, orth loss: %10.3e, relu loss: %10.3e | gradient: %10.3e"%(i, loss_at_it[0], loss_at_it[1], loss_at_it[2], loss_at_it[3], loss_at_it[4], gradient))
    
        # print_sim_of_all(I_target, I_source, p, output, pose_scale, sample_rate, spacing, poses, I_warped=I_warped)

    I_opt = p.detach().clone()
    proj_opt = output.detach().clone()
    del total_loss
    del loss
    del sim_loss_3d
    del model
    return I_opt, proj_opt, loss_log


def reconstruct_unconstrained(I_proj, I_target, poses, spacing, I_source = None, 
                              log_step=100, I_seg=None, I_warped=None, I_proj_orth=None, 
                              log_path = "./log", result_path="./result", pose_scale=""):
    
    device = torch.device("cuda" if use_cuda else "cpu")
    mu = 5
    eps = 1e-02
    I_init = torch.zeros(I_target.shape, device=device)
    lam = np.array([1.,1.,1.,1.,1.,1.])*0.1
    prev_loss = 1e-2
    lr_init = 8e-04

    

    # For ploting purpose, keep a numpy version of the data
    I_target_npy = I_target.cpu().numpy()[0,0]
    if I_warped is not None:
        I_warped_npy = I_warped.cpu().numpy()[0,0]
    else: 
        I_warped_npy = None

    iterations = 15
    for iter in range(iterations):
        print("Starting %i th lagrangian iteration................................."%iter)
        print("Mu is %f"%mu)
        print("Lambda are %f, %f, %f, %f, %f, %f"%(lam[0], lam[1], lam[2], lam[3], lam[4], lam[5]))
        print("Eps is %f"%eps)

        I_opt, proj_opt, loss_log = reconstruct(I_proj, I_target, poses, spacing, 
                                                I_init, mu, lam, eps, I_source, 
                                                epochs = 80, log_step=log_step, 
                                                I_seg = I_seg, I_warped = I_warped, 
                                                I_proj_orth = I_proj_orth, 
                                                log_path = log_path, 
                                                result_path = result_path, 
                                                pose_scale = pose_scale,
                                                lr_init=lr_init)

        # Converge test
        cur_loss = loss_log[-1][0]
        if abs((cur_loss-prev_loss)/prev_loss) <1e-4:
            break
        else: 
            prev_loss = cur_loss

        # Update lamda
        # lam[4] = lam[4] - mu*sim(torch.sum(I_opt,axis=4), I_proj_orth, reduction="mean")
        # lam[5] = lam[5] - mu*torch.sum((I_opt*I_seg)**2)
        # lam[5] = lam[5] - mu*torch.sum(F.relu(-1*I_opt))
        # lam[0] = lam[0] - mu*sim(proj_opt, I_proj)
        # lam[0] = lam[0] - mu*sim(torch.sum(I_opt,axis=2), I_proj, reduction="mean")
        for i in range(4):
            lam[i] = lam[i] - mu*sim(proj_opt[i], I_proj[i], 'mean')

        # Choose new mu, update new I_init and update eps
        mu = 1.1*mu
        I_init = I_opt.clone()
        eps = 0.5*eps
        lr_init = lr_init*0.4

        # Show current result
        plot_rec_result(I_opt.cpu().numpy()[0,0], I_target_npy,
                        save_path=os.path.join(log_path, "reconstructed_result_%i.png"%iter),
                        I_warped=I_warped_npy)
        
        print_sim_of_all(I_target, I_source, I_opt, proj_opt, pose_scale, sample_rate = [int(1), int(1), int(1)], 
                        spacing=spacing, poses=poses, I_warped=I_warped, 
                        save_path=os.path.join(log_path, "sim_all.csv"))
    np.save(result_path, I_opt.cpu().numpy())
    
        