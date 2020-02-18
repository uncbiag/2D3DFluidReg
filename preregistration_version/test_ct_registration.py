import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import mermaid.example_generation as EG
import mermaid.module_parameters as pars
import Similarity_measure
from preprocessing import preprocessData, calculate_projection, smoother

import mermaid
from mermaid import multiscale_optimizer as MO
from mermaid import utils as MUtils
import mermaid.module_parameters as pars
import os

from evaluate_dir_lab import eval_with_data, readPoint, eval_with_file

from Similarity_measure import SdtCTProjectionSimilarity

import warnings
warnings.filterwarnings("ignore")

# Synthesize projection angel
poses = np.array([#[-1., 2., -1.],
                  [-0.6, 2., -0.8],
                  [-0.3, 2., -0.5],
                  #[-0.2, 2., -0.2],
                  [-0.1, 2., -0.1],
                  [0., 2., 0.],
                  [0.1, 2., 0.1],
                  #[0.2, 2., 0.2],
                  [0.3, 2., 0.5],
                  [0.6, 2., 0.8]])
                  #[1., 2., 1.]])


#############################
# Load Params
path = "./lung_registration_setting.json"
lung_reg_params = pars.ParameterDict()
lung_reg_params.load_JSON(path)

torch.autograd.set_detect_anomaly(True)
#############################
# Data Preprocessing
resolution_scale = 2.
new_spacing = [1.5, 1.5, 1.5]
shape = [121, 512, 512]
spacing = [1., 1., 1.]
# spacing = [2.5, 0.625, 0.625]
sample_rate = [int(1), int(1), int(1)]
preprocessed_folder = lung_reg_params["preprocessed_folder"]
if lung_reg_params["recompute_preprocessed"] and \
   os.path.exists(preprocessed_folder + "/I0_3d.npy") and \
   os.path.exists(preprocessed_folder + "/I1_proj.npy"):
    preprocessData(lung_reg_params["source_file"],
                   lung_reg_params["target_file"],
                   preprocessed_folder,
                   shape,
                   spacing,
                   new_spacing,
                   smooth=True,
                   calc_projection=False,
                   poses=poses,
                   resolution_scale=resolution_scale,
                   sample_rate=sample_rate,
                   show_projection=False)


#############################
# Load CT data
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Source data

I0_numpy = np.load(preprocessed_folder+'/I0_3d.npy')
I0 = torch.from_numpy(I0_numpy).unsqueeze(0).unsqueeze(0).to(device)
# I0 = torch.from_numpy(smoother(I0_numpy, sigma=3)).unsqueeze(0).unsqueeze(0).to(device)


# Target data
I1_numpy = np.load(preprocessed_folder+'/I1_3d.npy')
I1 = torch.from_numpy(I1_numpy).unsqueeze(0).unsqueeze(0).to(device)
# I1 = torch.from_numpy(smoother(I1_numpy, sigma=3)).unsqueeze(0).unsqueeze(0).to(device)


# I1 = torch.from_numpy(np.load(lung_reg_params["affine_warped_file"])).to(device)
# proj = calculate_projection(np.load(lung_reg_params["affine_warped_file"])[0,0],
#                             poses,
#                             resolution_scale,
#                             sample_rate,
#                             device)
# proj = calculate_projection(I1_numpy,
#                             poses,
#                             resolution_scale,
#                             sample_rate,
#                             device)
# I1_proj = torch.from_numpy(proj).unsqueeze(0).to(device)
I1_proj = torch.from_numpy(np.load(preprocessed_folder+'/I1_proj.npy')).unsqueeze(0).to(device)

prop = np.load(preprocessed_folder + "/prop.npy",allow_pickle = True)
origin = np.flip(prop.item().get("crop")[0:3])
croped_dim = np.flip(prop.item().get("dim"))


#################################
# Performing the affine registration
sz = np.array(I0.shape).astype(np.int16)
mermaid_spacing = 1./(sz[2:] -1)

if lung_reg_params["affine"]["run"] :

    mermaid_params_affine = pars.ParameterDict()
    mermaid_params_affine.load_JSON(lung_reg_params["affine"]["setting"])

    affine_opt = MO.SimpleSingleScaleRegistration(I0,
                                                I1_proj,
                                                mermaid_spacing,
                                                sz,
                                                mermaid_params_affine,
                                                compute_inverse_map=True)

    affine_opt.get_optimizer().set_model(mermaid_params_affine['model']['registration_model']['type'])
    affine_opt.get_optimizer().add_similarity_measure("projection", Similarity_measure.SdtCTProjectionSimilarity)

    affine_opt.get_optimizer().set_visualization(True)
    affine_opt.get_optimizer().set_visualize_step(50)
    affine_opt.register()

    disp_map = affine_opt.get_map().detach()

    np.save(lung_reg_params["affine"]["disp_file"], disp_map.cpu().numpy())
    np.save(lung_reg_params["affine"]["warped_file"], affine_opt.get_warped_image().detach().cpu().numpy())
    inverse_map = MUtils.apply_affine_transform_to_map_multiNC(MUtils.get_inverse_affine_param(affine_opt.get_optimizer().model.Ab),
                                                            affine_opt.get_initial_inverse_map()).detach()
    # print(affine_opt.get_optimizer().model.Ab)
    np.save(lung_reg_params["affine"]["disp_inverse_file"], inverse_map.cpu().numpy())

    eval_with_file("../eval_data/copd1/copd1/copd1_300_iBH_xyz_r1.txt",
                "../eval_data/copd1/copd1/copd1_300_eBH_xyz_r1.txt",
                lung_reg_params["affine"]["disp_inverse_file"],
                croped_dim.copy(),
                np.flip(np.array(spacing)).copy(),
                origin,
                False)


# Performing the lddmm registration
if lung_reg_params["projection"]["run"]:

    mermaid_params_proj = pars.ParameterDict()
    mermaid_params_proj.load_JSON(lung_reg_params["projection"]["setting"])

    disp_map = torch.from_numpy(np.load(lung_reg_params["affine"]["disp_file"])).to(device)
    inverse_map = torch.from_numpy(np.load(lung_reg_params["affine"]["disp_inverse_file"])).to(device)

    mermaid_params_proj['model']['registration_model']['emitter_pos_list'] = poses.tolist()
    mermaid_params_proj['optimizer']['single_scale']['nr_of_iterations'] = 100
    mermaid_params_proj['model']['registration_model']['type'] = "lddmm_shooting_map" # "svf_vector_momentum_map"#"lddmm_shooting_map"
    mermaid_params_proj['model']['registration_model']['similarity_measure']['sigma'] = 0.01
    # mermaid_params_proj['model']['registration_model']['similarity_measure']['type'] = 'ncc'
    opt = MO.SimpleMultiScaleRegistration(I0,
                                        I1_proj,
                                        mermaid_spacing,
                                        sz,
                                        mermaid_params_proj,
                                        compute_inverse_map=True)

    opt.get_optimizer().set_model(mermaid_params_proj['model']['registration_model']['type'])
    opt.get_optimizer().add_similarity_measure("projection", Similarity_measure.SdtCTProjectionSimilarity)
    # opt.get_optimizer().set_initial_map(disp_map, map0_inverse = inverse_map)

    opt.get_optimizer().set_visualization(True)
    opt.get_optimizer().set_visualize_step(20)
    opt.register()

    # # opt.get_optimizer().save_checkpoint("./data/checkpoint")

    # ###############################
    # # Save the results
    np.save(lung_reg_params["projection"]["disp_file"], opt.get_map().detach().cpu().numpy())
    np.save(lung_reg_params["projection"]["warped_file"], opt.get_warped_image().detach().cpu().numpy())
    np.save(lung_reg_params["projection"]["disp_inverse_file"], opt.get_inverse_map().detach().cpu().numpy())
    # mermaid_params.write_JSON(lung_reg_params["mermaid_setting_file"])

    eval_with_file("../eval_data/copd1/copd1/copd1_300_iBH_xyz_r1.txt",
                "../eval_data/copd1/copd1/copd1_300_eBH_xyz_r1.txt",
                lung_reg_params["projection"]["disp_inverse_file"],
                croped_dim.copy(),
                np.flip(np.array(spacing)).copy(),
                origin,
                False)


