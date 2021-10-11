import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class projection(nn.Module):
    def __init__(self, shape, resolution, sample_rate, spacing, I_source, I_seg=None, device='cpu'):
        """
        :param shape: shape of the reconstructed image.
        :param resolution: resolution of the projectors.
        :param sample_rate: sample rate of the projectors.
        :param spacing: spacing of the reconstructed image.
        :param I_source: source image in BxCxDxWxH tensor format.
        :param I_seg: segmentation mask of the target image in BxCxDxWxH tensor format. Optional.
        """
        super(projection, self).__init__()
        if I_source is not None:
            self.I_rec = nn.Parameter(I_source.clone(), requires_grad=True)
        else:
            self.I_rec = nn.Parameter(torch.ones(shape).double()*0.01, requires_grad=True)

        self.resolution = resolution
        self.sample_rate = sample_rate
        self.spacing = torch.from_numpy(np.array(spacing)).to(device)
        self.batch_size = 4
        self.batch_start = 0
        self.I_seg = I_seg

    def forward(self, poses):
        grids, dx = self._project_grid_multi(poses, self.resolution,
                                             self.sample_rate,
                                             self.I_rec.shape[2:],
                                             self.spacing,
                                             self.I_rec.device)
        grids = torch.flip(grids, [4])
        (p, d, h, w) = grids.shape[0:4]
        b = self.I_rec.shape[0]
        grids = torch.reshape(grids, (1,1,1,-1,3))
        # dx = dx.unsqueeze(1).unsqueeze(1)
        projections = torch.mul(torch.sum(F.grid_sample(self.I_rec, grids.double(), align_corners = True).reshape((b, p, d, h, w)), dim=4), dx).float()

        # projections = torch.zeros((len(poses), 1, int(self.resolution[0]*self.sample_rate[0]), int(self.resolution[1]*self.sample_rate[2]))).to(self.I_rec.device)
        # for i in range(len(poses)):
        #     grid, dx = self.project_grid(self.I_rec, poses[i], (self.resolution[0], self.resolution[1]), self.sample_rate, self.I_rec.shape[2:])
        #     grid = torch.flip(grid, [3])
        #     dx = dx.unsqueeze(0).unsqueeze(0)
        #     if self.I_seg is not None:
        #         projections[i, 0] = torch.mul(torch.sum(F.grid_sample(self.I_rec*self.I_seg, grid.unsqueeze(0), align_corners=True), dim=4), dx)[0, 0]
        #     else: 
        #         projections[i, 0] = torch.mul(torch.sum(F.grid_sample(self.I_rec, grid.unsqueeze(0), align_corners=True), dim=4), dx)[0, 0]
        #     del grid
        #     torch.cuda.empty_cache()
            
        return projections

    def _project_grid_multi(self, emi_pos, resolution, sample_rate, obj_shape, spacing, device):
        # Axes definition: 0-axial, 1-coronal, 2-sagittal
        # sample_rate: sample count per pixel
        d, w, h = obj_shape
        (res_d, res_h) = resolution
        sr_d, sr_w, sr_h = sample_rate

        # P0 - one point in each coronal plane of CT. We use the points at Y axies.
        # I0 - start point of each rays.
        # N - Normal vector of coronal plane.
        P0 = torch.mm(
            torch.linspace(0, w-1, sr_w*w, device=device).unsqueeze(1), 
            torch.tensor([[0., 1., 0.]]).to(device))
        I0 = torch.from_numpy(emi_pos).to(device).float().unsqueeze(1).unsqueeze(1)
        N = torch.tensor([0., 1., 0.], device=device)
        
        # Calculate direction vectors for each rays
        lin_x = torch.linspace(-res_d/2, res_d/2-1, steps=res_d*sr_d)
        lin_y = torch.linspace(-res_h/2, res_h/2-1, steps=res_h*sr_h)
        grid_x, grid_y = torch.meshgrid(lin_x, lin_y)
        I = torch.zeros((lin_x.shape[0], lin_y.shape[0], 3), device=device)
        I[:,:,0] = grid_x
        I[:,:,2] = grid_y
        I = torch.add(I,-I0)
        dx = torch.mul(I, 1./I[:,:,:,1:2])
        I = I/torch.norm(I, dim=3, keepdim=True)
        dx = torch.norm(dx*spacing.to(device).unsqueeze(0).unsqueeze(0), dim=3)

        # Define a line as I(t)=I0+t*I
        # Define a plane as (P-P0)*N=0, P is a vector of points on the plane
        # Thus at the intersection of the line and the plane, we have d=(P0-I0)*N/(I*N)
        # Then we can get the position of the intersection by I(t) = t*I + I0
        # T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2), torch.matmul(P0-I0, N).unsqueeze(0))
        # grid = torch.add(torch.matmul(T.unsqueeze(3), I.unsqueeze(2)), I0)

        T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(3).unsqueeze(4), torch.matmul(P0-I0, N).unsqueeze(1).unsqueeze(1))
        grid = torch.add(torch.matmul(I.unsqueeze(4), T).permute(0,1,2,4,3), I0.unsqueeze(1))

        # Since grid_sample function accept input in range (-1,1)
        grid[:,:,:,:,0] = grid[:,:,:,:,0]/obj_shape[0]*2.0
        grid[:,:,:,:,1] = (grid[:,:,:,:,1]-0.)/obj_shape[1]*2.0 + -1.
        grid[:,:,:,:,2] = grid[:,:,:,:,2]/obj_shape[2]*2.0
        return grid, dx

    # Experiment for approach of applying filter to I_rec
    def transformed_I_rec(self):
        return torch.log((1+self.I_rec)/(1-self.I_rec))*0.5