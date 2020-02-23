# the parameter module which will keep track of all the registration parameters
import mermaid.module_parameters as pars
# and some mermaid functionality to create test data
import mermaid.example_generation as EG
import mermaid
from mermaid import multiscale_optimizer as MO

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def project_grid(img, emi_pos, resolution, sample_rate, obj_shape, spacing):
    d, w, h = obj_shape
    res_d, res_h = resolution
    device = img.device
    emi_pos_s = emi_pos / spacing
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
    return grid

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # #############################
    # # Load Params
    path = "./step_by_step_basic_settings.json"
    params = pars.ParameterDict()
    params.load_JSON(path)

    # #############################
    # # Load CT data
    preprocessed_file_folder = '../../Data/Preprocessed/sdt0001'
    case_pixels = np.load(preprocessed_file_folder+'/ct.npy')
    I0 = torch.from_numpy(case_pixels).unsqueeze(0).unsqueeze(0)
    spacing = np.array([1., 1., 1.])
    sampler = mermaid.image_sampling.ResampleImage()
    I0, spacing = sampler.downsample_image_by_factor(I0, spacing, scalingFactor=1.)
    I0 = I0.to(device)
    # I0 = I0.permute(0,1,4,3,2)
    #spacing = np.array([1., 1., 1.])


    # ###############################
    # # Get sDT Projections
    projectionPos_file_path = '../../Data/Raw/NoduleStudyProjections/001/projectionPos.csv'
    projectionDicom_path = '../../Data/Raw/NoduleStudyProjections/001/DICOM'
    poses = pd.read_csv(projectionPos_file_path).to_numpy()
    theta_list = [get_theta_tensor(p) for p in poses]
    case = np.load(preprocessed_file_folder+'/projection.npy').astype(np.float32)
    case_torch = torch.from_numpy(case).unsqueeze(1).to(device)
    case_torch, spac = sampler.downsample_image_by_factor(case_torch, np.array([1., 1.]), scalingFactor=0.15)
    I1 = case_torch.permute(1,0,2,3)
    I1 = I1.to(device)

    #################################################
    # For toy demo input
    #################################################
    # I0,I1_3d,spacing = EG.CreateSquares(dim=3,add_noise_to_bg=False).create_image_pair(np.array([64,64,64]),params=params)
    # I1_3d[:, :, 32:, 32:, 32:] = 0
    # I0 = torch.from_numpy(I0).to(device)
    # I1_3d = torch.from_numpy(I1_3d).to(device)

    # spacing = np.array([1.,1.,1.])
    # poses = np.array([[-32,100,0],[0,100,0]])

    # poses_ct_space = poses
    # I0 = I1_3d


    poses_ct_space = np.zeros(poses.shape, dtype=np.float32)
    poses_ct_space[:,0] = poses[:,1]
    poses_ct_space[:,1] = poses[:,2]
    poses_ct_space[:,2] = poses[:,0]

    proj_w = int(I1.shape[2])
    proj_h = int(I1.shape[3])
    I0_pro = torch.ones((1, poses_ct_space.shape[0], proj_w, proj_h)).to(device)
    for i in range(poses_ct_space.shape[0]):
        grid = torch.flip(project_grid(I0, poses_ct_space[i,:], (proj_w, proj_h), I0.shape[2:], spacing),[3])
        temp = F.grid_sample(I0, grid.unsqueeze(0))
        I0_pro[0, i] = (torch.sum(F.grid_sample(I0, grid.unsqueeze(0)), dim=4)[0,0])

    fig, axs = plt.subplots(1,4)
    axs[0].imshow(I0_pro[0,1].detach().cpu().numpy(), cmap="gray")
    axs[1].imshow(I0_pro[0,10].detach().cpu().numpy(), cmap="gray")
    axs[2].imshow(I1[0,1].detach().cpu().numpy())
    axs[3].imshow(I1[0,10].detach().cpu().numpy())

    plt.show()
    # plt.savefig("./data/test_projection.png")
