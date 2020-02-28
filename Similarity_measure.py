from mermaid.similarity_measure_factory import SimilarityMeasure, NCCSimilarity, LNCCSimilarity, SSDSimilarity, OptimalMassTransportSimilarity
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from evaluate_dir_lab import eval_with_data, readPoint

class ProjectionSimilarity(SimilarityMeasure):
    """
    Computes a projection based similarity measure between two images.
    """
    def __init__(self, spacing, params):
        super(ProjectionSimilarity,self).__init__(spacing,params)

    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (NCC)/sigma^2
       """
        sim_x = F.mse_loss(torch.sum(I0,2),torch.sum(I1,2))
        sim_y = F.mse_loss(torch.sum(I0,3),torch.sum(I1,3))
        return sim_x+sim_y

class SdtCTParallelProjectionSimilarity(SimilarityMeasure):
    """
    Computes a projection based similarity measure between two images.
    """
    def __init__(self, spacing, params):
        super(SdtCTParallelProjectionSimilarity,self).__init__(spacing,params)
        self.nccSim = NCCSimilarity(spacing[0::2], params)

    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (NCC)/sigma^2
       """
        sim = 0.
        theta_list = self.params['theta_list']
        resolution = self.params["projection_resolution"]
        resolution = torch.Size([1,1,resolution[0],resolution[1], resolution[1]])
        for i in range(len(theta_list)):
            proj = self.project_diagonal(I0, torch.tensor([theta_list[i]]).to(I0.device), resolution)
            # sim += 1 - self.ncc_sqr(proj, I1[:,i:i+1,:,:])
            sim += self.nccSim.compute_similarity(proj, I1[:,i:i+1,:,:])
            # sim += F.mse_loss(proj,I1[:,i,:,:])
        return sim/len(theta_list)/self.sigma**2
    
    def ncc_sqr(self, I0, I1):
        I0_dev = I0-torch.mean(I0)
        I1_dev = I1-torch.mean(I1)
        return (torch.sum(I0_dev*I1_dev)**2)/(torch.sum(I0_dev**2)*torch.sum(I1_dev**2))

    def project(self, img):
        return torch.sum(img,dim=3)

    def project_diagonal(self,img, delta_theta_x, resolution):
        device = img.device
        d = img.shape[2]
        half_d = int(np.ceil(d/4))
        # img = torch.ones(img.shape).to(device) - img
        # new_img = torch.nn.functional.pad(img, (half_d, half_d, half_d, half_d, half_d, half_d), 'constant', 0)
        # theta_z = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).to(device)*torch.cos(delta_theta_z)+\
        #         torch.Tensor([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).to(device)*torch.sin(delta_theta_z)
        theta_x = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).to(device)*torch.cos(delta_theta_x)+\
                torch.Tensor([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]).to(device)*torch.sin(delta_theta_x)
        # theta = torch.mm(theta_z, theta_x)

        aff_grid = F.affine_grid(theta_x[0:3, :].unsqueeze(0), resolution)
        return torch.sum(F.grid_sample(img, aff_grid, padding_mode="zeros"), 3)

class SdtCTProjectionSimilarity(SimilarityMeasure):
    """
    Computes a projection based similarity measure between two images.
    """
    def __init__(self, spacing, params):
        super(SdtCTProjectionSimilarity,self).__init__(spacing,params)
        self.use_normalized_gradient = False
        self.use_omt = False
        if self.params['similarity_measure']["projection"]["base"] == "ssd":
            self.sim = SSDSimilarity(spacing[0::2], params)
        elif self.params["similarity_measure"]["projection"]["base"] == "ncc":
            self.sim = NCCSimilarity(spacing[0::2], params)
        elif self.params["similarity_measure"]["projection"]["base"] == "lncc":
            self.sim = LNCCSimilarity(spacing[0::2], params)
        elif self.params["similarity_measure"]["projection"]["base"] == "omt":
            self.sim = OptimalMassTransportSimilarity(spacing[0::2], params)
            self.use_omt = True
        elif self.params["similarity_measure"]["projection"]["base"] == "ngf":
            self.sim = None
            self.use_normalized_gradient = True
        else:
            self.sim = SSDSimilarity(spacing[0::2], params)
        self.emi_poses = self.params['emitter_pos_list']/spacing+1
        self.proj_res_scale = self.params['resolution_scale']
        self.sample_rate = self.params["sample_rate"]
        self.grids = None
        self.x_gradient_filter = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view(1,1,3,3)
        self.y_gradient_filter = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3)
        self.volumeElement = spacing[0]*spacing[2]
        self.sigma = self.params['similarity_measure']['sigma']

    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (NCC)/sigma^2
       """
        sim = 0.
        proj_sim = 0.
        grad_sim = 0.
        proj_res = [I1.shape[2], I1.shape[3]]

        # idx = torch.multinomial(torch.ones([len(self.emi_poses)]), 3)

        # Version 1. Compute grids every iteration
        for i in range(len(self.emi_poses)):
            grids, dx = self._project_grid(self.emi_poses[i], proj_res, self.sample_rate, I0.shape[2:], self.spacing, I0.device)
            grids = torch.flip(grids, [3]).unsqueeze(0)
            dx = dx.unsqueeze(0).unsqueeze(0)
            I0_proj = torch.mul(torch.sum(F.grid_sample(I0, grids, align_corners = False), dim=4), dx)
            if self.sim and not self.use_normalized_gradient:
                if self.use_omt:
                    proj_sim = proj_sim + self.sim.compute_similarity(I0_proj[0,0], I1[0,i,:,:], I0_proj, phi)
                else:
                    proj_sim = proj_sim + self.sim.compute_similarity(I0_proj, I1[:,i:i+1,:,:], I0_proj, phi)
            else:
                # Calculate gradient similarity
                g_I0 = self._image_gradient(I0_proj, I0.device)
                g_I1 = self._image_gradient(I1[:,i:i+1,:,:], I0.device)
                volumeElement = I1.shape[2]*I1.shape[3]
                # temp = F.cosine_similarity(g_I0, g_I1, dim=4, eps=1e-16)
                cross = torch.bmm(g_I0.view(-1,1,2), g_I1.view(-1,2,1)).view(-1,1) + 1e-9
                norm = torch.mul(torch.norm(g_I0.view(-1,2),dim=1), 
                                 torch.norm(g_I1.view(-1,2),dim=1)) + 1e-9
                sim_per_pix = 1. - torch.div(cross.view(-1), norm)
                grad_sim = torch.sum(sim_per_pix)/volumeElement/self.sigma
                proj_sim = proj_sim + grad_sim

        sim = proj_sim

        # # TODO: debug
        # (h, w) = g_I0.shape[2:4]
        # g_I0_np = g_I0[0].detach().cpu().numpy()
        # g_I1_np = g_I1[0].detach().cpu().numpy()
        # X,Y=np.meshgrid(np.arange(0,h,1),np.arange(0,w,1))
        # fig, axes = plt.subplots(3,7)
        # for i in range(3):
        #     index = i*3
        #     axes[i,0].imshow(I0_proj[0,index].detach().cpu().numpy())
        #     axes[i,0].set_title("I0 projection")

        #     axes[i,1].imshow(I1[0,index].detach().cpu().numpy())
        #     axes[i,1].set_title("I1 projection")

        #     axes[i,2].imshow((I0_proj[0,index]-I1[0,index]).detach().cpu().numpy())
        #     # axes[i,2].imshow(g_I0_np[index,:,:,0])
        #     # axes[i,2].set_title("I0 gradient along x")

        #     g = axes[i,3].imshow(g_I1_np[index,:,:,0])
        #     axes[i,3].set_title("I1 gradient along x")
        #     fig.colorbar(g, ax = axes[i,3])

        #     axes[i,4].imshow(g_I0_np[index,:,:,1])
        #     axes[i,4].set_title("I0 gradient along y")

        #     axes[i,5].imshow(g_I1_np[index,:,:,1])
        #     axes[i,5].set_title("I1 gradient along y")

        #     diff = axes[i,6].imshow(sim_per_pix.view(I1.shape[1],
        #                                       I1.shape[2], 
        #                                       I1.shape[3]
        #                                       )[index,:,:].detach().cpu().numpy())
        
        #     fig.colorbar(diff, ax = axes[i,6])

        # # plt.show()
        # plt.savefig("./log/gradient_1.png")

        # fig, axes = plt.subplots(3,1)
        # for i in range(3):
        #     axes[i].imshow(temp.view(11,I1.shape[2], I1.shape[3])[i*2,:,:].detach().cpu().numpy())

        # plt.savefig("./log/gradient_dif.png")


        return sim/len(self.emi_poses)

    def project(self, I0, emi_poses, proj_res, sample_rate):
        grid, dx = self._project_grid(emi_poses, proj_res, sample_rate, I0.shape[2:], self.spacing, I0.device)
        grid = torch.flip(grid, [3]).unsqueeze(0)
        dx = dx.unsqueeze(0).unsqueeze(0)
        proj = torch.mul(torch.sum(F.grid_sample(I0, grid, align_corners = False), dim=4), dx)
        del grid
        return proj

    def _image_gradient(self, x, device):
        new_x = x.permute(1,0,2,3)
        x_filter = self.x_gradient_filter.to(device)
        g_x = F.conv2d(new_x,x_filter, padding=1)

        y_filter = self.y_gradient_filter.to(device)
        g_y = F.conv2d(new_x,y_filter, padding=1)

        temp = torch.pow(g_x,2) + torch.pow(g_y,2)
        g = torch.stack([g_x,g_y])/torch.sqrt(torch.pow(g_x,2) + torch.pow(g_y,2) + 0.1)
        return g.permute(2,1,3,4,0)

    def _project_grid(self, emi_pos, resolution, sample_rate, obj_shape, spacing, device):
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
        I0 = torch.from_numpy(emi_pos).to(device).float()
        N = torch.tensor([0., 1., 0.], device=device)
        
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
        # T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2), torch.matmul(P0-I0, N).unsqueeze(0))
        # grid = torch.add(torch.matmul(T.unsqueeze(3), I.unsqueeze(2)), I0)

        T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2).unsqueeze(3), torch.matmul(P0-I0, N).unsqueeze(0))
        grid = torch.add(torch.matmul(I.unsqueeze(3), T).permute(0,1,3,2), I0)

        # Since grid_sample function accept input in range (-1,1)
        grid[:,:,:,0] = grid[:,:,:,0]/obj_shape[0]*2.0
        grid[:,:,:,1] = (grid[:,:,:,1]-0.)/obj_shape[1]*2.0 + -1.
        grid[:,:,:,2] = grid[:,:,:,2]/obj_shape[2]*2.0
        return grid, dx

