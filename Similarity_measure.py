from mermaid.similarity_measure_factory import SimilarityMeasure, NCCSimilarity, LNCCSimilarity
import torch
import torch.nn.functional as F
import numpy as np

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
        self.nccSim = LNCCSimilarity(spacing[0::2], params) #LNCCSimilarity(spacing, params)
        self.emi_poses = self.params['emitter_pos_list']/spacing
        self.proj_res = self.params['projection_resolution']
        self.sample_rate = self.params["sample_rate"]
        self.grids = None
        self.x_gradient_filter = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view(1,1,3,3)
        self.y_gradient_filter = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view(1,1,3,3)

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

        #Calculate projection
        if self.grids is None:
            self.grids = torch.zeros([len(self.emi_poses), 1, self.proj_res[0]*self.sample_rate[0], self.proj_res[1]*self.sample_rate[1], I0.shape[2], 3],device=I0.device)
            for i in range(len(self.emi_poses)):
                self.grids[i] = torch.flip(self._project_grid(self.emi_poses[i], self.proj_res, self.sample_rate, I0.shape[2:], self.spacing, I0.device), [3]).unsqueeze(0)

        I0_proj = torch.empty_like(I1)

        for i in range(len(self.emi_poses)):
            # proj = torch.sum(F.grid_sample(I0, grid.unsqueeze(0)), dim=4)
            I0_proj[:,i,:,:] = torch.sum(F.grid_sample(I0, self.grids[i], align_corners = False), dim=4)

            # TODO: workaround for lncc. Remove later.
            proj_double = I0_proj[:,i,:,:].double()
            I1_double = I1[:,i:i+1,:,:].double()
            proj_sim += self.nccSim.compute_similarity(proj_double, I1_double)

        #Calculate gradient similarity
        g_I0 = self._image_gradient(I0_proj, I0.device)
        g_I1 = self._image_gradient(I1, I0.device)
        grad_sim = self.nccSim.compute_similarity(g_I0, g_I1)

        sim = proj_sim + grad_sim

        return sim/len(self.emi_poses)/self.sigma**2

    def project(self, I0, emi_poses, proj_res, sample_rate):
        grid = torch.flip(self._project_grid(emi_poses, proj_res, sample_rate, I0.shape[2:], self.spacing, I0.device), [3])
        proj = torch.sum(F.grid_sample(I0, grid.unsqueeze(0), align_corners = True), dim=4)
        del grid
        return proj

    def _image_gradient(self, x, device):
        x_filter = self.x_gradient_filter.repeat(1, x.shape[1], 1, 1).to(device)
        g_x = F.conv2d(x,x_filter)

        y_filter = self.y_gradient_filter.repeat(1, x.shape[1], 1, 1).to(device)
        g_y = F.conv2d(x,y_filter)

        g = torch.sqrt(torch.pow(g_x,2)+torch.pow(g_y,2))
        return g

    def _project_grid(self, emi_pos, resolution, sample_rate, obj_shape, spacing, device):
        # Axes definition: 0-axial, 1-coronal, 2-sagittal
        # sample_rate: sample count per pixel
        d, w, h = obj_shape
        res_x, res_y = resolution
        sr_x, sr_y = sample_rate

        # P0 - one point in each coronal plane of CT. We use the points at Y axies.
        # I0 - start point of each rays.
        # N - Normal vector of coronal plane.
        P0 = torch.mm(
            torch.linspace(0, d-1, d, device=device).unsqueeze(1), 
            torch.tensor([[0., 1., 0.]]).to(device))
        I0 = torch.from_numpy(emi_pos).to(device).float()
        N = torch.tensor([0., 1., 0.], device=device)
        
        # Calculate direction vectors for each rays
        lin_x = torch.linspace(-res_x/2, res_x/2-1, steps=res_x*sr_x)
        lin_y = torch.linspace(-res_y/2, res_y/2-1, steps=res_y*sr_y)
        grid_x, grid_y = torch.meshgrid(lin_x, lin_y)
        I = torch.zeros((lin_x.shape[0], lin_y.shape[0], 3), device=device)
        I[:,:,0] = grid_x
        I[:,:,2] = grid_y
        I = torch.add(I,-I0)
        I = I/torch.norm(I, dim=2, keepdim=True)

        # Define a line as I(t)=I0+t*I
        # Define a plane as (P-P0)*N=0, P is a vector of points on the plane
        # Thus at the intersection of the line and the plane, we have d=(P0-I0)*N/(I*N)
        # Then we can get the position of the intersection by I(t) = t*I + I0
        T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2), torch.matmul(P0-I0, N).unsqueeze(0))
        grid = torch.add(torch.matmul(T.unsqueeze(3), I.unsqueeze(2)), I0)

        # Since grid_sample function accept input in range (-1,1)
        grid[:,:,:,0] = grid[:,:,:,0]/obj_shape[0]*2.0
        grid[:,:,:,1] = (grid[:,:,:,1]-0.)/obj_shape[1]*2.0 + -1.
        grid[:,:,:,2] = grid[:,:,:,2]/obj_shape[2]*2.0
        return grid

