import mermaid.example_generation as EG
import mermaid.module_parameters as pars
import mermaid.registration_networks as RN

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import functional as F

params = pars.ParameterDict()
use_synthetic_test_case = True

# Desired dimension (mermaid supports 1D, 2D, and 3D registration)
dim = 2

# If we want to add some noise to the background (for synthetic examples)
add_noise_to_bg = True

# and now create it
if use_synthetic_test_case:
    length = 64
    # size of the desired images: (sz)^dim
    szEx = np.tile(length, dim )
    # create a default image size with two sample squares
    I0, I1, spacing = EG.CreateSquares(dim,add_noise_to_bg).create_image_pair(szEx, params)
else:
    # return a real image example
    I0, I1, spacing = EG.CreateRealExampleImages(dim).create_image_pair() 

# fig, axes = plt.subplots(1,2)
# axes[0].imshow(I0[0,0,:,:])
# axes[1].imshow(I1[0,0,:,:])
# plt.show()

def project(img, axis=0):
    return torch.sum(img, axis)


def project_diagonal(img, delta_theta):
    device = img.device
    d = img.shape[2]
    half_d = int(np.ceil(d/2))
    #img = torch.ones(img.shape).to(device) - img
    new_img = torch.nn.functional.pad(img, (half_d, half_d, half_d, half_d), 'constant', 1)
    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device)*torch.cos(delta_theta)\
         + torch.Tensor([[0, 1, 0], [-1, 0, 0]]).to(device)*torch.sin(delta_theta)

    aff_grid = F.affine_grid(theta.unsqueeze(0), new_img.shape)
    return torch.sum(F.grid_sample(new_img, aff_grid, padding_mode="zeros"), 2)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

projection_theta = torch.tensor([45./180.]).to(device)

I1_torch = torch.from_numpy(I1).to(device)
target_projection = torch.cat([project(I1_torch, 2), project(I1_torch, 3)], dim=0)
target_projection_dia = project_diagonal(I1_torch, projection_theta*np.pi)

num_iterations = int(2e3)


def optimizer_grad_step(optimizer):
    for opt in optimizer:
        opt.step()
        opt.zero_grad()
    return optimizer


optimizer = []
# For three direction projection with 3e3 iterations
optimizer.append(optim.Adam([
    {'params':source_trans, 'init_lr':0.1, 'lr_decay':6000, 'lr':0.1},
    {'params': source_rotation, 'init_lr':0.3, 'lr_decay':2000, 'lr': 0.3}
], betas=(0.7, 0.999)))
loss = torch.nn.MSELoss()
loss_dia = torch.nn.MSELoss(reduction="sum")

from mermaid.utils import utils
from mermaid.data_wrapper import AdaptVal
import mermaid.model_factory as MF
# Create the identity map
id = utils.identity_map_multiN(sz, spacing)
identityMap = AdaptVal(torch.from_numpy(id))

sz = I0.shape
spacing = I0.shape
mf = MF.ModelFactory(sz, spacing, sz, spacing)
model, criterion = mf.create_registration_model("svf_vector_momentum_map", none, compute_inverse_map=compute_inverse_map)

error = 0.
for iteration in range(num_iterations):

    for opt in optimizer:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group["init_lr"]*(0.1**(iteration/param_group['lr_decay']))

    # Apply registration transformer
    source_img, phi_warped, phi_inverseWarped = evaluate_model_low_level_interface(model,I0)
    
    
    # Calc Projection
    source_projection = torch.cat([project(source_img, 2), project(source_img, 3)], dim=0)
    source_projection_dia = project_diagonal(source_img, projection_theta*np.pi)
    
    # output = loss(source_projection_dia, target_projection_dia)
    output = (loss(source_projection[0], target_projection[0])\
             # + loss(source_projection[1],target_projection[1])\
             + loss(source_projection_dia,target_projection_dia))/2
    output.backward()
    optimizer_grad_step(optimizer)

    # compare learned volume to data generating one
    error += output.detach().cpu().numpy()
    print("Iteration {}/{}, error: {:.3f}".format(iteration, num_iterations, output), end='\n', flush=True)
    

#########################Show Result##############################
print("target translation: ", target_trans.cpu().numpy())
print("learned translation: ", source_trans.detach().cpu().numpy())
print("target rotation: ", target_rotation.cpu().numpy())
print("learned rotation: ", source_rotation.detach().cpu().numpy())


def showImg(ax_list, data, projection_x, projection_y, projection_dia, title, title_setting):
    ax_list[0, 0].imshow(data)
    ax_list[0, 0].set_title(title, title_setting)

    d = projection_x.shape[0]
    ax_list[1, 0].bar(np.linspace(0, d-1, d), projection_x-np.ones(d)*d, width=1,  align="edge")
 
    ax_list[0, 1].barh(np.linspace(0, d-1, d), np.ones(d)*d-projection_y, height=1, align="edge")
    ax_list[0, 1].invert_yaxis()

    d = projection_dia.shape[0]
    ax_list[1, 1].bar(np.linspace(0, d-1, d), projection_dia, width=1, align="edge")

fig, axes = plt.subplots(2, 6, figsize=(12, 4))

source_img = translate(I0.unsqueeze(0), source_trans, source_rotation*np.pi)
source_projection = torch.cat([project(source_img, 2), project(source_img, 3)], dim=0)
source_projection_dia = project_diagonal(source_img, projection_theta*np.pi)
title_setting = {"fontsize": 6}
plt.rc('xtick', labelsize=2)
plt.rc('ytick', labelsize=2)

showImg(axes[:, 0:2], img_torch.cpu().numpy(), 
    project(img_torch, 0).cpu().numpy(), 
    project(img_torch, 1).cpu().numpy(), 
    project_diagonal(img_torch.repeat(1, 1, 1, 1),
    projection_theta*np.pi)[0, 0, :].cpu().numpy(), "Source image", title_setting)

# image rotation
showImg(axes[:, 2:4], source_img.detach().cpu().numpy()[0, 0, :, :]\
    ,source_projection[0, 0, :].detach().cpu().numpy()\
    ,source_projection[1, 0, :].detach().cpu().numpy()\
    ,source_projection_dia[0, 0, :].detach().cpu().numpy()\
    ,"Registered image", title_setting)

# image skew
showImg(axes[:, 4:6], target_img[0, 0, :,:].cpu().numpy()\
    ,target_projection[0, 0, :].cpu().numpy()\
    ,target_projection[1, 0, :].cpu().numpy()\
    ,target_projection_dia[0, 0, :].cpu().numpy()\
    ,"Target image", title_setting)

plt.show()
