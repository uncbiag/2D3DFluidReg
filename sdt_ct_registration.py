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

import Similarity_measure

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#############################
# Load Params
path = "./step_by_step_basic_settings.json"
params = pars.ParameterDict()
params.load_JSON(path)

#############################
# Load CT data
preprocessed_file_folder = '../../Data/Preprocessed/sdt0001'
case_pixels = np.load(preprocessed_file_folder+'/ct.npy')
I0 = torch.from_numpy(case_pixels).unsqueeze(0).unsqueeze(0)
spacing = np.array([1., 1., 1.])
sampler = mermaid.image_sampling.ResampleImage()
I0, spacing = sampler.downsample_image_by_factor(I0, spacing, scalingFactor=0.2)
I0 = I0.to(device)


###############################
# Get sDT Projections

projectionPos_file_path = '../../Data/Raw/NoduleStudyProjections/001/projectionPos.csv'
projectionDicom_path = '../../Data/Raw/NoduleStudyProjections/001/DICOM'
poses = pd.read_csv(projectionPos_file_path).to_numpy()
poses_ct_space = np.zeros(poses.shape, dtype=np.float32)
poses_ct_space[:,0] = poses[:,1]
poses_ct_space[:,1] = poses[:,2]
poses_ct_space[:,2] = poses[:,0]
poses_ct_space = poses_ct_space/spacing

case = np.load(preprocessed_file_folder+'/projection.npy').astype(np.float32)
case_torch = torch.from_numpy(case).unsqueeze(1).to(device)
case_torch, spac = sampler.downsample_image_by_factor(case_torch, np.array([1., 1.]), scalingFactor=0.03)
I1 = case_torch.permute(1,0,2,3)
I1 = I1.to(device)

###############################
# Set params
params['model']['registration_model']['emitter_pos_list'] = poses_ct_space.tolist()
params['model']['registration_model']['projection_resolution'] = [int(I1.shape[2]), int(I1.shape[3])]
params['model']['registration_model']['sample_rate'] = [int(1), int(1)]
#params['optimizer']['sgd']['individual']['lr'] = 0.1
params['optimizer']['single_scale']['nr_of_iterations'] = 10
params['model']['registration_model']['similarity_measure']['type'] = "sdtctprojection"
print(params)

#################################################
# For toy demo input
#################################################
# def project_diagonal(img, delta_theta_x):
#     theta_x = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).to(device)*torch.cos(delta_theta_x)+\
#             torch.Tensor([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]).to(device)*torch.sin(delta_theta_x)

#     aff_grid = F.affine_grid(theta_x[0:3, :].unsqueeze(0), img.shape)
#     return torch.sum(F.grid_sample(img, aff_grid, padding_mode="zeros"), 3)

# I0,I1_3d,spacing = EG.CreateSquares(dim=3,add_noise_to_bg=False).create_image_pair(np.array([64,64,64]),params=params)
# I1_3d = I0.copy()
# I1_3d = -np.log(I1_3d+1.3)
# I1_3d[:, :, 15:30, 15:30, 15:30] = I1_3d[:, :, 5:20, 5:20, 5:20]
# I0 = torch.from_numpy(I0).to(device)
# I1_3d = torch.from_numpy(I1_3d).to(device)

# theta_list = [0., 0.1*np.pi, -0.1*np.pi]
# I1 = torch.ones((1, len(theta_list), I0.shape[2], I0.shape[4])).to(device)
# for i in range(len(theta_list)):
#     I1[0, i] = project_diagonal(I1_3d, torch.tensor((theta_list[i])))[0, 0]

# params.load_JSON('step_by_step_example_data.json')
# params['model']['registration_model']['similarity_measure']['type'] = "SdtCTParallelProjection"
# params['model']['registration_model']['theta_list'] = theta_list


#################################
# Performing the registration
sz = np.array(I0.shape)
spacing = np.array([1., 1., 1.])
# params['model']['registration_model']['type'] = "affine_map"
# params['model']['registration_model']['similarity_measure']['sigma'] = 1.
# affine_opt = MO.SimpleSingleScaleRegistration(I0,
#                                                    I1,
#                                                    spacing,
#                                                    sz,
#                                                    params,
#                                                    compute_inverse_map=False)
# affine_opt.get_optimizer().set_visualization(True)
# affine_opt.get_optimizer().set_visualize_step(5)
# affine_opt.register()

# I0 = affine_opt.get_warped_image().detach()

params['model']['registration_model']['type'] = "svf_vector_momentum_map"
params['model']['registration_model']['similarity_measure']['sigma'] = 0.1
opt = MO.SimpleSingleScaleRegistration(I0,
                                       I1,
                                       spacing,
                                       sz,
                                       params,
                                       compute_inverse_map=False)

opt.get_optimizer().set_model(params['model']['registration_model']['type'])
opt.get_optimizer().add_similarity_measure("projection", Similarity_measure.SdtCTProjectionSimilarity)

opt.get_optimizer().set_visualization(True)
opt.get_optimizer().set_visualize_step(20)
opt.register()

###############################
# Save the results
np.save("./data/input_noMask_large.npy", I0.detach().cpu().numpy())
np.save("./data/disp_noMask_large.npy", opt.get_map().detach().cpu().numpy())
np.save("./data/warped_noMask_large.npy", opt.get_warped_image().detach().cpu().numpy())
params.write_JSON(path)

