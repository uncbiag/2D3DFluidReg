from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

img = Image.open("./img.png")
img_torch = transforms.ToTensor()(img)
img_torch = img_torch[0].to(device)

# plt.imshow(img_torch.numpy())
# plt.show()

def translate(img_torch,delt_dist, delta_theta):
    device = img_torch.device
    #delt_dist_inrange = torch.fmod(delt_dist,0.7)
    #delta_theta_inrange = torch.fmod(delta_theta,2)

    #Fistly, move the image to the center
    theta_1 = torch.Tensor([[1,0,0],[0,1,0]]).to(device)*torch.cos(delta_theta) + torch.Tensor([[0,1,0],[-1,0,0]]).to(device)*torch.sin(delta_theta)
    theta_2 = torch.cat([torch.Tensor([[1,0],[0,1],[0,0]]).to(device),torch.cat([delt_dist,torch.ones(1,1).to(device)])],dim=-1).to(device)
    theta = torch.mm(theta_1,theta_2)

    grid = F.affine_grid(theta.unsqueeze(0),img_torch.unsqueeze(0).size())
    output = F.grid_sample(img_torch.unsqueeze(0),grid,padding_mode="border")
    return output

def project(img,axis=0):
    return torch.sum(img,axis)

def project_diagonal(img):
    d = img.shape[2]
    idx = torch.linspace(1-d,d-1,2*d-1)
    idx = idx.type(torch.int)
    return torch.Tensor([torch.sum(torch.ones(torch.diagonal(img[0,0,:,:],i).shape[0])-torch.diagonal(img[0,0,:,:],i))  for i in idx])
    

target_trans = torch.rand(2,1) #-1~1
target_rotation = torch.rand(1) #-1~1
target_img = translate(img_torch.unsqueeze(0),target_trans,target_rotation*np.pi)
target_projection = torch.cat([project(target_img,2),project(target_img,3)],dim=0)
target_projection_dia = project_diagonal(target_img)

source_trans = nn.Parameter((torch.zeros(2,1)).to(device),requires_grad=True)
source_rotation = nn.Parameter((torch.zeros(1)).to(device),requires_grad=True)

num_iterations = int(2e3)

def optimizer_grad_step(optimizer):
    for opt in optimizer:
        opt.step()
        opt.zero_grad()
    return optimizer


optimizer = []
init_lr = 0.1
lr_decay = 1000

# For three direction projection with 3e3 iterations
optimizer.append(optim.Adam([
    {'params':source_trans,'init_lr':0.1,'lr_decay':6000,'lr':0.1},
    {'params': source_rotation, 'init_lr':0.4,'lr_decay':3000,'lr': 0.4}
], betas=(0.7, 0.999)))
loss = torch.nn.MSELoss()

error = 0.
for iteration in range(num_iterations):

    for opt in optimizer:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group["init_lr"]*(0.1**(iteration/param_group['lr_decay']))
    
    source_img = translate(img_torch.unsqueeze(0),source_trans,source_rotation*np.pi)
    source_projection = torch.cat([project(source_img,2),project(source_img,3)],dim=0)
    source_projection_dia = project_diagonal(source_img)
    
    
    output = (loss(source_projection[0],target_projection[0])\
             + loss(source_projection[1],target_projection[1])\
             + loss(source_projection_dia,target_projection_dia))/3
    output.backward()
    optimizer_grad_step(optimizer)

    # compare learned volume to data generating one
    error += output.detach().cpu().numpy()
    print("Iteration {}/{}, error: {:.3f}".format(iteration, num_iterations, output), end='\n', flush=True)
    
print("target translation: ",target_trans.numpy())
print("learned translation: ",source_trans.detach().numpy())
print("target rotation: ",target_rotation.numpy())
print("learned rotation: ",source_rotation.detach().numpy())

def showImg(ax_list, data, projection_x, projection_y, projection_dia, title, title_setting):
    ax_list[0,0].imshow(data)
    ax_list[0,0].set_title(title, title_setting)

    d = projection_x.shape[0]
    ax_list[1,0].bar(np.linspace(0,d-1,d), projection_x-np.ones(d)*d, width=1, align="edge")
 
    ax_list[0,1].barh(np.linspace(0,d-1,d), np.ones(d)*d-projection_y, height=1, align="edge")
    ax_list[0,1].invert_yaxis()
    
    d = projection_dia.shape[0]
    ax_list[1,1].bar(np.linspace(0,d-1,d), projection_dia, width=1, align="edge")

fig, axes = plt.subplots(2, 6,figsize=(12,4))
source_img = translate(img_torch.unsqueeze(0),source_trans,source_rotation*np.pi)
source_projection = torch.cat([project(source_img,2),project(source_img,3)],dim=0)
source_projection_dia = project_diagonal(source_img)
title_setting = {"fontsize":6}
plt.rc('xtick',labelsize=2)
plt.rc('ytick',labelsize=2)

showImg(axes[:,0:2], img_torch.numpy(), project(img_torch,0).numpy()
        ,project(img_torch,1).numpy(),project_diagonal(img_torch.repeat(1,1,1,1)),"Source image",title_setting)

# image rotation
showImg(axes[:,2:4], source_img.detach().numpy()[0,0,:,:], source_projection[0,0,:].detach().numpy()
        ,source_projection[1,0,:].detach().numpy(),source_projection_dia.detach().numpy(),"Registered image",title_setting)

# image skew
showImg(axes[:,4:6], target_img[0,0,:,:].numpy(), target_projection[0,0,:].numpy()
        ,target_projection[1,0,:].numpy(),target_projection_dia.numpy(),"Target image",title_setting)

plt.show()

