from preprocessing import calculate_projection
import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.functional import l1_loss
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
import mermaid.module_parameters as pars
import pydicom as dicom
import random
import os

import matplotlib.pyplot as plt

from CTPlayground import resample


class projection(nn.Module):
    def __init__(self, shape, resolution, sample_rate, I_target, spacing, I_source):
        super(projection, self).__init__()
        # self.I_rec = nn.Parameter(torch.rand_like(I_target, device=device), requires_grad=True)
        # self.I_rec = nn.Parameter(torch.ones(shape, device=device)*-50, requires_grad=True)
        
        if I_source is not None:
            self.I_rec = nn.Parameter(I_source.clone(), requires_grad=True)
        else:
            # rec = np.zeros(shape)
            # radius = (np.min(rec.shape[2:])/3)**2
            # mid = [int(i/2) for i in rec.shape[2:]]
            # for i in range(rec.shape[2]):
            #     for j in range(rec.shape[3]):
            #         for p in range(rec.shape[4]):
            #             if (i-mid[0])**2+(j-mid[1])**2+(p-mid[2])**2 < radius:
            #                 rec[0,0,i,j,p] = 0.01
            # rec = rec.astype(np.float32)
            # self.I_rec = nn.Parameter(torch.from_numpy(rec).to(device), requires_grad=True)
            self.I_rec = nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)


        self.resolution = resolution
        self.sample_rate = sample_rate
        self.spacing = torch.from_numpy(spacing).to(device)
        # self.spacing = torch.from_numpy(spacing).to(device)*0.001
        self.batch_size = 4
        self.batch_start = 0

    def forward(self, poses):
        # projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
        projections = torch.zeros((self.batch_size, 1, int(self.resolution[0]*self.sample_rate[0]), int(self.resolution[1]*self.sample_rate[2]))).to(device)
        idx = random.sample(list(range(poses.shape[0])), self.batch_size)
        # idx = range(0, poses.shape[0])
        # idx = range(self.batch_start, min(self.batch_start+self.batch_size, poses.shape[0]))
        # if self.batch_start+self.batch_size > poses.shape[0]:
        #     self.batch_start = 0
        # else:
        #     self.batch_start += self.batch_size
        for i in range(len(idx)):
            grid, dx = self.project_grid(self.I_rec, poses[idx[i]], (self.resolution[0], self.resolution[1]), self.sample_rate, self.I_rec.shape[2:])
            grid = torch.flip(grid, [3])
            dx = dx.unsqueeze(0).unsqueeze(0)
            projections[i, 0] = torch.mul(torch.sum(F.grid_sample(self.I_rec, grid.unsqueeze(0), align_corners=True), dim=4), dx)[0, 0]
            del grid
            torch.cuda.empty_cache()
            
        return projections, idx

    def project_grid(self, img, emi_pos, resolution, sample_rate, obj_shape):
        d, w, h = obj_shape
        res_d, res_h = resolution
        device = img.device
        emi_pos_s = emi_pos
        sr_d, sr_w, sr_h = sample_rate

        # P0 - one point in each coronal plane of CT. We use the points at Y axies.
        # I0 - start point of each rays.
        # N - Normal vector of coronal plane.
        P0 = torch.mm(torch.linspace(0,w-1,sr_w*w,device=device).unsqueeze(1), torch.tensor([[0., 1., 0.]]).to(device))
        I0 = torch.from_numpy(emi_pos_s).to(device).float()
        N = torch.tensor([0.,1.,0.], device=device)
        
        # Calculate direction vectors for each rays
        lin_x = torch.linspace(-res_d/2, res_d/2-1, steps=res_d*sr_d)
        lin_y = torch.linspace(-res_h/2, res_h/2-1, steps=res_h*sr_h)
        grid_x, grid_y = torch.meshgrid(lin_x, lin_y)
        I = torch.zeros((lin_x.shape[0], lin_y.shape[0], 3), device=device)
        I[:,:,0] = grid_x
        I[:,:,2] = grid_y
        I = torch.add(I,-I0)
        dx = torch.mul(I, 1./I[:,:,1:2])
        I = I/torch.norm(I, dim=2, keepdim=True)
        dx = torch.norm(dx*self.spacing.unsqueeze(0).unsqueeze(0), dim=2)
        # dx = torch.abs(torch.mul(torch.ones((I.shape[0],I.shape[1]), device=device),1./I[:,:,1]))

        # Define a line as I(t)=I0+t*I
        # Define a plane as (P-P0)*N=0, P is a vector of points on the plane
        # Thus at the intersection of the line and the plane, we have d=(P0-I0)*N/(I*N)
        # Then we can get the position of the intersection by I(t) = t*I + I0
        T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2).unsqueeze(3), torch.matmul(P0-I0, N).unsqueeze(0))
        grid = torch.add(torch.matmul(I.unsqueeze(3), T).permute(0,1,3,2), I0)

        # Since grid_sample function accept input in range (-1,1)
        grid[:,:,:,0] = grid[:,:,:,0]/obj_shape[0]*2.0
        grid[:,:,:,1] = (grid[:,:,:,1]-0.)/obj_shape[1]*2.0 + -1.
        grid[:,:,:,2] = grid[:,:,:,2]/obj_shape[2]*2.0
        return grid, dx

def image_gradient(x):
    device = x.device
    
    fil = torch.tensor([[1,2,1],[2,4,2],[1,2,1]])
    x_filter = torch.zeros((3,3,3), device=device).view(1,1,3,3,3)
    x_filter[0,0,0,:,:]=fil
    x_filter[0,0,2,:,:]=-fil
    g_x = F.conv3d(x, x_filter, padding=1)

    y_filter = torch.zeros((3,3,3), device=device).view(1,1,3,3,3)
    y_filter[0,0,:,0,:]=fil
    y_filter[0,0,:,2,:]=-fil
    g_y = F.conv3d(x, y_filter, padding=1)

    z_filter = torch.zeros((3,3,3), device=device).view(1,1,3,3,3)
    z_filter[0,0,:,:,0]=fil
    z_filter[0,0,:,:,2]=-fil
    g_z = F.conv3d(x, z_filter, padding=1)

    return torch.norm(torch.cat((g_x, g_y, g_z), 0), dim=0)

def reconstruct(I_proj, I_target, poses, spacing, I_source = None, epochs = 400, log_step=100, log_path = "./log", result_path="./result"):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    resolution = [I_proj.shape[2], I_proj.shape[3]]
    sample_rate = [int(1), int(1), int(1)]
    I_target_npy = I_target.cpu().numpy()[0,0]
    model = projection(I_target.shape, resolution, sample_rate, I_target, spacing, I_source)
    # opt = torch.optim.Adam(model.parameters(), lr=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    # opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(opt, 'min', factor=0.3, patience=2, cooldown=30, verbose=True)

    for i in range(epochs):
        opt.zero_grad()
        output, idx = model(poses)
        loss = l1_loss(output, I_proj[idx,:,:,:], reduction='mean')
        # loss = torch.mean(torch.norm(output-I_proj[idx,:,:,:], dim=(2,3)))
        total_loss = loss
        for p in model.parameters():
            # setting for CT
            total_loss = total_loss + 100*torch.mean(F.relu(-1*p)) + torch.mean(image_gradient(p))
            # setting for sDCT
            # total_loss = total_loss + 1000*torch.sum(F.relu(-1*p)) + torch.mean(image_gradient(p))
        #early stop
        if total_loss.data < 1e-4:
            break
        total_loss.backward()
        opt.step()
        scheduler.step(loss)


        if i%log_step == 0:
            for p in model.parameters():
                diff = l1_loss(p, I_target, reduction='mean')
                print("The L1 loss in 3d is: %s"%diff)

                reconstructed = p.detach().cpu().numpy()[0,0]
                step = int(reconstructed.shape[1]/5)

                fig, axs = plt.subplots(2,5)
                for j in range(0,5):
                    im1 = axs[0,j].imshow(reconstructed[:,j*step,:])
                    cb = plt.colorbar(im1, ax = axs[0,j], shrink=0.5, pad=0.02)
                    cb.ax.tick_params(labelsize=5)
                    im2 = axs[1,j].imshow(I_target_npy[:,j*step,:])
                    cb = plt.colorbar(im2, ax = axs[1,j], shrink=0.5, pad=0.02)
                    cb.ax.tick_params(labelsize=5)
                plt.savefig(os.path.join(log_path,"reconstructed_full_%i.jpg"%i), dpi=300)
                
                fig, axs = plt.subplots(2,4)
                for j in range(0,4):
                    im1 = axs[0,j].imshow(output[j,0].detach().cpu().numpy())
                    cb = plt.colorbar(im1, ax = axs[0,j], shrink=0.5, pad=0.02)
                    cb.ax.tick_params(labelsize=5)
                    im2 = axs[1,j].imshow(I_proj[idx,0,:,:][j].cpu().numpy())
                    cb = plt.colorbar(im2, ax = axs[1,j], shrink=0.5, pad=0.02)
                    cb.ax.tick_params(labelsize=5)
                plt.savefig(os.path.join(log_path,"projection_%i.jpg"%i), dpi=300)
            

        print("Epoch %i: total loss: %f, rec loss: %f"%(i, total_loss.data, loss.data))

    for p in model.parameters():
        # diff = l1_loss(p, I_target, reduction='mean')
        diff = torch.mean((p-I_target)**2)
        print("The MSE loss in 3d is: %s"%diff)

        reconstructed = p.detach().cpu().numpy()[0,0]
        step = int(reconstructed.shape[1]/5)

        np.save(result_path, reconstructed)

        fig, axs = plt.subplots(2,5)
        for j in range(0,5):
            axs[0,j].imshow(reconstructed[:,j*step,:])
            axs[1,j].imshow(I_target_npy[:,j*step,:])
        # plt.show()
        plt.savefig(os.path.join(log_path, "reconstructed_full.jpg"), dpi=300)


parser = argparse.ArgumentParser(description='3D/2D registration')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')
parser.add_argument('--data', default='dirlab')
parser.add_argument('--result','-r', default='./result')
parser.add_argument('--preprocess','-p',default="")
parser.add_argument('--exp','-e',default="exp")
parser.add_argument('--with_prior',type=int,default=0)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
args = parser.parse_args()

if args.data == "dirlab":
    #Prepare poses 
    poses = np.array([
                    [-0.3, 3., -0.2],
                    [-0.1, 3., -0.1],
                    [0.1, 3., 0.1],
                    [0.3, 3., 0.2]])
    # poses = np.ndarray((30,3),dtype=np.float)
    # poses[:,1] = 4.
    # poses[:,0] = np.linspace(-0.4,0.4, num = 30)
    # poses[:,2] = np.linspace(-0.2,0.2, num = 30)

    resolution_scale = 1.4
    new_spacing = [1., 1., 1.]
    sample_rate = [int(1), int(1), int(1)]

    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)
    if args.preprocess == "":
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    else:
        preprocessed_folder = args.preprocess
    prefix = lung_reg_params["source_img"].split("/")[-3]

    if args.with_prior == 1:
        I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I0_3d.npy')
        I_source = torch.from_numpy(I_numpy).unsqueeze(0).unsqueeze(0).to(device)
    else: 
        I_source = None
    I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d.npy')
    I_target = torch.from_numpy(I_numpy).unsqueeze(0).unsqueeze(0).to(device)

    poses = poses*I_numpy.shape[1]

    I_proj = torch.from_numpy(np.load(preprocessed_folder+'/' + prefix + '_I1_proj.npy')).unsqueeze(1).to(device)

    if not os.path.exists(args.result):
        os.mkdir(args.result)
    log_path = "./log/reconstruct_"+ args.data + "/" +args.exp +"/" + prefix
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    result_path = args.result+'/'+ prefix + ".npy"
    
    reconstruct(I_proj, I_target, poses, np.array(new_spacing), I_source, 800, 100, log_path, result_path)
elif args.data == "sDCT":
    projection_pos_file_path = '../../Data/Raw/NoduleStudyProjections/001/projectionPos.csv'
    projection_dicom_path = '../../Data/Raw/NoduleStudyProjections/001/DICOM'
    sDCT_dicom_path = "../../Data/Raw/DICOMforMN/S00002/SER00001"
    poses_origin = pd.read_csv(projection_pos_file_path).to_numpy()
    poses = np.zeros(poses_origin.shape, dtype=np.float32)
    poses[:,0] = poses_origin[:,1]
    poses[:,1] = poses_origin[:,2]
    poses[:,2] = poses_origin[:,0]
    current_spacing = np.array([0.139, 4, 0.139])
    new_spacing = np.array([1, 4, 1])
    scale_factor = current_spacing/new_spacing
    poses = poses/new_spacing

    # Load projections
    proj_file_list = os.listdir(projection_dicom_path)
    proj_file_list.sort()
    image = [dicom.read_file(projection_dicom_path + '/' + s).pixel_array for s in proj_file_list]
    image = np.array(image)
    image = image.astype(np.float32)/65535+0.0001
    I_proj = torch.from_numpy(image).unsqueeze(1).to(device)
    # # I_proj = F.interpolate(I_proj, scale_factor=scale_factor[0::2])
    # # I_proj = I_proj/(torch.max(I_proj)-torch.min(I_proj))
    # # I_proj[I_proj==0] = 0.1
    I_proj = - torch.log(F.interpolate(I_proj, scale_factor=scale_factor[0::2]))

    # plt.imshow(image[int(image.shape[0]/2),:,:],cmap='gray')
    # plt.axis("off")
    # plt.savefig("./figure/projection.jpg",dpi=300, bbox_inches="tight", pad_inches=0)

    # Load sDCT
    sdct_file_list = os.listdir(sDCT_dicom_path)
    sdct_file_list.sort()
    image  = [dicom.read_file(sDCT_dicom_path + '/' + s).pixel_array for s in sdct_file_list]
    image = np.array(image)
    image = np.transpose(image, (1,0,2))
    image = image.astype(np.float32)/100000.
    I_target = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    I_target = F.interpolate(I_target, scale_factor=scale_factor)

    prefix = projection_dicom_path.split("/")[-2]
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    log_path = "./log/reconstruct_"+ args.data + "/" +prefix
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # reconstruct(I_proj, I_target, poses, new_spacing, 1000, 100, log_path, args.result+'/' + prefix + "_rec.npy")
    # np.save(args.result+'/' + prefix + "_origin.npy", I_target[0,0].cpu().numpy())

# reconstructed = np.load(os.path.join("./log", "reconstruct_" + args.data + "/reconstructed_full.npy"))
# # I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d.npy')
# # reconstructed = reconstructed/(np.max(reconstructed)-np.min(reconstructed))*2000
# fig, axs = plt.subplots(2,5)
# fig.suptitle("Reconstructed result")
# for i in range(0,10):
#     axs[int(i/5),i%5].imshow(reconstructed[:,i*20,:])
#     axs[int(i/5),i%5].set_xlabel("i = %i"%(i*7))
# # plt.show()
# plt.savefig(os.path.join("./log", "reconstruct_" + args.data + "/reconstructed_full_sparse.jpg"), dpi=300)

# I_numpy = I_target[0,0].cpu().numpy()
# fig, axs = plt.subplots(2,5)
# fig.suptitle("sDCT")
# for i in range(0,10):
#     axs[int(i/5),i%5].imshow(I_numpy[:,i*20,:])
#     axs[int(i/5),i%5].set_xlabel("i = %i"%(i*7))
#     # axs[1,j].imshow(I_numpy[:,100+j*10,:])
# # plt.show()
# plt.savefig(os.path.join("./log", "reconstruct_" + args.data + "/sDCT.jpg"), dpi=300)

# # Plot the hist
# fig, axs = plt.subplots(3,2)
# fig.suptitle("Histagram")
# axs[0,0].hist(reconstructed[:,35,:]*1000, bins=100)
# axs[0,0].set_ylabel("reconstructed")
# axs[0,1].imshow(reconstructed[:,35,:])
# axs[1,0].hist(I_numpy[:,35,:]*1000, bins=100)
# axs[1,0].set_ylabel("sDCT")
# axs[1,1].imshow(I_numpy[:,35,:])
# axs[2,1].imshow(I_numpy[:,35,:]-reconstructed[:,35,:])
# plt.savefig(os.path.join("./log", "reconstruct_" + args.data + "/hist.jpg"), dpi=300)