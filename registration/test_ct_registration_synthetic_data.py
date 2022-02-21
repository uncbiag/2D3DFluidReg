import os

import mermaid
import mermaid.example_generation as EG
import mermaid.module_parameters as pars
import numpy as np
import torch
from mermaid import multiscale_optimizer as MO
from tools.preprocessing import calculate_projection

import registration.similarity_measure as similarity_measure

# Synthesize projection angel
poses = np.array([#[-0.6, 2., -0.06],
                    #[-0.5, 2., -0.05],
                    [-0.3, 2., 0.04],
                    [-0.2, 2., 0.03],
                    [-0.1, 2., 0.02],
                    [0., 2., 0.],
                    [0.1, 2., 0.02],
                    [0.2, 2., 0.03],
                    #[0.3, 2., 0.04],
                    #[0.5, 2., 0.05],
                    [0.6, 2., 0.06]])


#############################
# Load Params
path = "./lung_registration_setting.json"
lung_reg_params = pars.ParameterDict()
lung_reg_params.load_JSON(path)

mermaid_params = pars.ParameterDict()
mermaid_params.load_JSON(lung_reg_params["mermaid_setting_file"])

#############################
# Pytorch setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.autograd.set_detect_anomaly(True)

#############################
# Data Prepare
mermaid_params["square_example_images"]["len_s"] = 20
mermaid_params["square_example_images"]["len_l"] = 20
I0_3d, I1_3d, spacing = EG.CreateSquares(dim=3, add_noise_to_bg=False).create_image_pair(np.array([64, 64, 64]), params=mermaid_params)
resolution_scale = 1.5
sample_rate = [int(1), int(4), int(1)]

I0_3d[:, :, 26:36, 26:36, 25:36] = 0
I1_3d[:, :, 26:36, 20:36, 25:36] = 0

np.save(lung_reg_params["source_file_synthetic"], I0_3d[0,0])
np.save(lung_reg_params["target_file_synthetic"], I1_3d[0,0])

I0 = torch.from_numpy(I0_3d).to(device)

proj = calculate_projection(I1_3d[0, 0],
                            poses,
                            resolution_scale,
                            sample_rate,
                            device)
I1_proj = torch.from_numpy(proj).unsqueeze(0).to(device)


###############################
# Set mermaid_params
mermaid_params['model']['registration_model']['emitter_pos_list'] = poses.tolist()
mermaid_params['model']['registration_model']['projection_resolution'] = [int(I0.shape[2]*resolution_scale), int(I0.shape[4]*resolution_scale)]
mermaid_params['model']['registration_model']['sample_rate'] = sample_rate
mermaid_params['optimizer']['single_scale']['nr_of_iterations'] = 200

mermaid_params['optimizer']['name'] = 'lbfgs_ls' #'lbfgs_ls'
# mermaid_params['optimizer']['sgd']['individual']['lr'] = 0.000001
mermaid_params['optimizer']['scheduler']['patience'] = 3
# mermaid_params['model']['registration_model']['similarity_measure']['type'] = 'lncc'



#################################
# Performing the affine registration
sz = np.array(I0.shape)
mermaid_spacing = 1.0/sz[2:]

# mermaid_params['model']['registration_model']['type'] = "affine_map"
# mermaid_params['model']['registration_model']['similarity_measure']['sigma'] = 0.1
# affine_opt = MO.SimpleSingleScaleRegistration(I0,
#                                               I1,
#                                               mermaid_spacing,
#                                               sz,
#                                               mermaid_params,
#                                               compute_inverse_map=False)

# # affine_opt.get_optimizer().set_model(mermaid_params['model']['registration_model']['type'])
# # affine_opt.get_optimizer().add_similarity_measure("projection", Similarity_measure.SdtCTProjectionSimilarity)

# affine_opt.get_optimizer().set_visualization(False)
# affine_opt.get_optimizer().set_visualize_step(10)
# affine_opt.register()

# disp_map = affine_opt.get_map().detach()

# np.save(lung_reg_params["affine_disp_file"], disp_map.cpu().numpy())
# np.save(lung_reg_params["affine_warped_file"], affine_opt.get_warped_image().detach().cpu().numpy())

# Performing the lddmm registration

# disp_map = torch.from_numpy(np.load(lung_reg_params["affine_disp_file"])).to(device)

mermaid_params['optimizer']['single_scale']['nr_of_iterations'] = 50
mermaid_params['model']['registration_model']['type'] = "svf_vector_momentum_map" # "svf_vector_momentum_map"#"lddmm_shooting_map"
# mermaid_params['model']['registration_model']['similarity_measure']['sigma'] = 0.01
mermaid_params['model']['registration_model']['similarity_measure']['sigma'] = 0.05
mermaid_params['optimizer']['name'] = 'lbfgs_ls'
opt = MO.SimpleSingleScaleRegistration(I0,
                                       I1_proj,
                                       mermaid_spacing,
                                       sz,
                                       mermaid_params,
                                       compute_inverse_map=False)

opt.get_optimizer().set_model(mermaid_params['model']['registration_model']['type'])
opt.get_optimizer().add_similarity_measure("projection", similarity_measure.SdtCTProjectionSimilarity)
# opt.get_optimizer().set_initial_map(disp_map)

opt.get_optimizer().set_visualization(True)
opt.get_optimizer().set_visualize_step(10)
opt.register()

# opt.get_optimizer().save_checkpoint("./data/checkpoint")

###############################
# Save the results
np.save(lung_reg_params["disp_file"], opt.get_map().detach().cpu().numpy())
np.save(lung_reg_params["warped_file"], opt.get_warped_image().detach().cpu().numpy())
# mermaid_params.write_JSON(lung_reg_params["mermaid_setting_file"])


