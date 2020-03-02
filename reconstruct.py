from preprocessing import calculate_projection
import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import l1_loss
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
import mermaid.module_parameters as pars
import random

import matplotlib.pyplot as plt


class projection(nn.Module):
    def __init__(self, I_target):
        super(projection, self).__init__()
        # self.I_rec = nn.Parameter(torch.rand_like(I_target, device=device), requires_grad=True)
        self.I_rec = nn.Parameter(torch.ones_like(I_target, device=device), requires_grad=True)
        # self.I_rec = nn.Parameter(I_target.clone(), requires_grad=True)

    def forward(self, poses, resolution_scale, sample_rate):
        resolution = [int(self.I_rec.shape[2] * resolution_scale),
                    int(self.I_rec.shape[4] * resolution_scale)]
        # projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
        projections = torch.zeros((1, 10, resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
        idx = random.sample(list(range(poses.shape[0])),10)
        for i in range(len(idx)):
            grid, dx = self.project_grid(self.I_rec, poses[idx[i]], (resolution[0], resolution[1]), sample_rate, self.I_rec.shape[2:])
            grid = torch.flip(grid, [3])
            dx = dx.unsqueeze(0).unsqueeze(0)
            projections[0, i] = torch.mul(torch.sum(F.grid_sample(self.I_rec, grid.unsqueeze(0), align_corners=False), dim=4), dx)[0, 0]
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
        I = I/torch.norm(I, dim=2, keepdim=True)
        dx = torch.abs(torch.mul(torch.ones((I.shape[0],I.shape[1]), device=device),1./I[:,:,1]))

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

parser = argparse.ArgumentParser(description='3D/2D registration')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')

poses = np.array([
                  [-0.4, 3., -0.5],
                  [-0.1, 3., -0.1],
                  [0.1, 3., 0.1],
                  [0.3, 3., 0.5]])
poses = np.ndarray((30,3),dtype=np.float)
poses[:,1] = 4.
poses[:,0] = np.linspace(-0.4,0.4, num = 30)
poses[:,2] = np.linspace(-0.2,0.2, num = 30)

resolution_scale = 1.4
new_spacing = [1., 1., 1.]
sample_rate = [int(1), int(1), int(1)]

args = parser.parse_args()
lung_reg_params = pars.ParameterDict()
lung_reg_params.load_JSON(args.setting)
preprocessed_folder = lung_reg_params["preprocessed_folder"]
prefix = lung_reg_params["source_img"].split("/")[-3]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d.npy')
I_target = torch.from_numpy(I_numpy).unsqueeze(0).unsqueeze(0).to(device)

# reconstructed = np.load("./reconstructed.npy")
# fig, axes = plt.subplots(3,10)
# for i in range(0,30):
#     axes[int(i/10),i%10].imshow(reconstructed[:,100+i*4,:])
# plt.savefig("./log/temp.jpg")

poses = poses*I_numpy.shape

I_proj = torch.from_numpy(np.load(preprocessed_folder+'/' + prefix + '_I1_proj.npy')).unsqueeze(0).to(device)

model = projection(I_target)
opt = torch.optim.Adam(model.parameters(), lr=0.5)
scheduler = ReduceLROnPlateau(opt, 'min', factor=0.3, patience=2, cooldown=30, verbose=True)

log_step = 100

# for i in range(400):
#     opt.zero_grad()
#     output, idx = model(poses, resolution_scale, sample_rate)
#     loss = l1_loss(output, I_proj[:,idx,:,:], reduction='mean')
#     total_loss = loss
#     for p in model.parameters():
#         total_loss = total_loss + torch.mean(image_gradient(p))
#     #early stop
#     if total_loss.data < 1e-4:
#         break
#     total_loss.backward()
#     opt.step()
#     scheduler.step(loss)

#     if i%log_step == 0:
#         for p in model.parameters():
#             diff = l1_loss(p, I_target, reduction='sum')
#             print("The L1 loss in 3d is: %s"%diff)

#             reconstructed = p.detach().cpu().numpy()[0,0]

#             fig, axs = plt.subplots(2,5)
#             for j in range(0,5):
#                 axs[0,j].imshow(reconstructed[:,100+j*10,:])
#                 axs[1,j].imshow(I_numpy[:,100+j*10,:])
#             # plt.show()
#             plt.savefig("./log/reconstructed_full_%i.jpg"%i, dpi=300)
        

#     print("Epoch %i: total loss: %f, rec loss: %f"%(i, total_loss.data, loss.data))

# for p in model.parameters():
#     diff = l1_loss(p, I_target, reduction='sum')
#     print("The L1 loss in 3d is: %s"%diff)

#     reconstructed = p.detach().cpu().numpy()[0,0]

#     np.save("./log/reconstructed_full.npy", reconstructed)

#     fig, axs = plt.subplots(2,5)
#     for j in range(0,5):
#         axs[0,j].imshow(reconstructed[:,100+j*10,:])
#         axs[1,j].imshow(I_numpy[:,100+j*10,:])
#     # plt.show()
#     plt.savefig("./log/reconstructed_full.jpg", dpi=300)


reconstructed = np.load("./log/reconstructed_full.npy")
I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d.npy')
fig, axs = plt.subplots(2,5)
for i in range(0,10):
    axs[int(i/5),i%5].imshow(reconstructed[:,50+i*25,:])
    # axs[1,j].imshow(I_numpy[:,100+j*10,:])
# plt.show()
plt.savefig("./log/reconstructed/reconstructed_full_sparse.jpg", dpi=300)
for i in range(0,10):
    axs[int(i/5),i%5].imshow(I_numpy[:,30+i*25,:])
    # axs[1,j].imshow(I_numpy[:,100+j*10,:])
# plt.show()
plt.savefig("./log/reconstructed/ct_.jpg", dpi=300)