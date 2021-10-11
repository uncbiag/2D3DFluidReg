# the parameter module which will keep track of all the registration parameters
import mermaid.module_parameters as pars
# and some mermaid functionality to create test data
import mermaid.example_generation as EG
import mermaid
from mermaid import multiscale_optimizer as MO
from mermaid import utils as MUtils

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os

from Similarity_measure import SdtCTProjectionSimilarity
import argparse

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load Params
parser = argparse.ArgumentParser(description='3D/2D registration')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')   
parser.add_argument('--preprocess', '-p', metavar='PREPROCESS', default='',
                    help='Path of the folder contains preprocess files.')


def main(args):
    #############################
    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)

    exp_path = args.disp_f
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    prefix="sDCT"

    if args.preprocess == "":
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    else:
        preprocessed_folder = args.preprocess
    # Load Params
    # path = "./step_by_step_basic_settings.json"
    # params = pars.ParameterDict()
    # params.load_JSON(path)

    #############################
    # Load CT data
    case_pixels = np.load(preprocessed_folder+'/ct.npy')
    I0 = torch.from_numpy(case_pixels).unsqueeze(0).unsqueeze(0)
    spacing = np.array([1., 1., 1.])
    shape = I0.shape[2:]
    sampler = mermaid.image_sampling.ResampleImage()
    I0, spacing = sampler.downsample_image_by_factor(I0, spacing, scalingFactor=0.6)
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
    poses_ct_space = poses_ct_space/shape[1]

    case = np.load(preprocessed_folder+'/projection.npy').astype(np.float32)
    case_torch = torch.from_numpy(case).unsqueeze(1).to(device)
    case_torch, spac = sampler.downsample_image_by_factor(case_torch, np.array([1., 1.]), scalingFactor=0.1)
    I1 = case_torch.permute(1,0,2,3)
    I1_proj = I1.to(device)

    ###############################
    # Set params
    # params['model']['registration_model']['emitter_pos_list'] = poses_ct_space.tolist()
    # params['model']['registration_model']['projection_resolution'] = [int(I1.shape[2]), int(I1.shape[3])]
    # params['model']['registration_model']['sample_rate'] = [int(1), int(1)]
    # #params['optimizer']['sgd']['individual']['lr'] = 0.1
    # params['optimizer']['single_scale']['nr_of_iterations'] = 10
    # params['model']['registration_model']['similarity_measure']['type'] = "sdtctprojection"
    # print(params)

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
    # Performing the affine registration
    sz = np.array(I0.shape).astype(np.int16)
    mermaid_spacing = 1./(sz[2:] - 1)

    if lung_reg_params["affine"]["run"]:

        mermaid_params_affine = pars.ParameterDict()
        mermaid_params_affine.load_JSON(lung_reg_params["affine"]["setting"])
        mermaid_params_affine['model']['registration_model']['emitter_pos_scale_list'] = poses_ct_space.tolist()
        mermaid_params_affine['model']['registration_model']["similarity_measure"]["projection"]["spacing"] = spacing
        mermaid_params_affine['model']['registration_model']["similarity_measure"]["projection"]["currentScaleFactor"] = 1.0

        affine_opt = MO.SimpleMultiScaleRegistration(I0,
                                                    I1_proj,
                                                    mermaid_spacing,
                                                    sz,
                                                    mermaid_params_affine,
                                                    compute_inverse_map=True)

        affine_opt.get_optimizer().set_model(mermaid_params_affine['model']['registration_model']['type'])
        affine_opt.get_optimizer().add_similarity_measure("projection", SdtCTProjectionSimilarity)

        affine_opt.get_optimizer().set_visualization(False)
        affine_opt.get_optimizer().set_visualize_step(50)
        affine_opt.get_optimizer().set_save_fig(True)
        affine_opt.get_optimizer().set_expr_name("3d_2d_sDCT")
        affine_opt.get_optimizer().set_save_fig_path("./log")
        affine_opt.get_optimizer().set_pair_name(["affine"])

        affine_opt.register()

        disp_map = affine_opt.get_map().detach()
        np.save(os.path.join(exp_path, prefix + "_affine_disp.npy"), disp_map.cpu().numpy())
        np.save(os.path.join(exp_path, prefix + "_affine_warped.npy"), affine_opt.get_warped_image().detach().cpu().numpy())
        inverse_map = MUtils.apply_affine_transform_to_map_multiNC(MUtils.get_inverse_affine_param(affine_opt.get_optimizer().ssOpt.model.Ab),
                                                                affine_opt.get_optimizer().ssOpt.get_initial_inverse_map()).detach()
        # inverse_map = MUtils.apply_affine_transform_to_map_multiNC(MUtils.get_inverse_affine_param(affine_opt.get_optimizer().model.Ab),
        #                                                         affine_opt.get_initial_inverse_map()).detach()
        # print(affine_opt.get_optimizer().model.Ab)
        np.save(os.path.join(exp_path, prefix + "_affine_inverse_disp.npy"), inverse_map.cpu().numpy())

        del affine_opt
        torch.cuda.empty_cache()

    # Performing the lddmm registration
    if lung_reg_params["deformable"]["run"]:

        mermaid_params_proj = pars.ParameterDict()
        mermaid_params_proj.load_JSON(lung_reg_params["deformable"]["setting"])

        try:
            disp_map = torch.from_numpy(np.load(os.path.join(exp_path, prefix + "_affine_disp.npy"))).to(device)
            inverse_map = torch.from_numpy(np.load(os.path.join(exp_path, prefix + "_affine_inverse_disp.npy"))).to(device)
        except:
            disp_map = None
            inverse_map = None
            print("Did not find affine disp map.")

        mermaid_params_proj['model']['registration_model']['emitter_pos_scale_list'] = poses_ct_space.tolist()
        mermaid_params_proj['optimizer']['single_scale']['nr_of_iterations'] = 5
        mermaid_params_proj['model']['registration_model']['type'] = "lddmm_shooting_map" # "svf_vector_momentum_map"#"lddmm_shooting_map"
        mermaid_params_proj['model']['registration_model']['similarity_measure']['sigma'] = 0.1
        mermaid_params_proj['model']['registration_model']["similarity_measure"]["projection"]["spacing"] = spacing
        mermaid_params_proj['model']['registration_model']["similarity_measure"]["projection"]["currentScaleFactor"] = 1.0
        # mermaid_params_proj['model']['registration_model']['similarity_measure']['type'] = 'ncc'
        opt = MO.SimpleMultiScaleRegistration(I0,
                                            I1_proj,
                                            mermaid_spacing,
                                            sz,
                                            mermaid_params_proj,
                                            compute_inverse_map=True)

        opt.get_optimizer().set_model(mermaid_params_proj['model']['registration_model']['type'])
        opt.get_optimizer().add_similarity_measure("projection", SdtCTProjectionSimilarity)
        if lung_reg_params['deformable']['use_affine'] and (disp_map is not None) and (inverse_map is not None):
            opt.get_optimizer().set_initial_map(disp_map, map0_inverse = inverse_map)

        opt.get_optimizer().set_visualization(False)
        opt.get_optimizer().set_visualize_step(20)
        opt.get_optimizer().set_save_fig(True)
        opt.get_optimizer().set_expr_name("3d_2d_sDCT")
        opt.get_optimizer().set_save_fig_path("./log")
        opt.get_optimizer().set_pair_name(["lddmm"])

        opt.register()

        # # opt.get_optimizer().save_checkpoint("./log/checkpoint")

        # ###############################
        # # Save the results
        np.save(os.path.join(exp_path, prefix + "_lddmm_disp.npy"), opt.get_map().detach().cpu().numpy())
        np.save(os.path.join(exp_path, prefix + "_lddmm_warped.npy"), opt.get_warped_image().detach().cpu().numpy())
        np.save(os.path.join(exp_path, prefix + "_lddmm_inverse_disp.npy"), opt.get_inverse_map().detach().cpu().numpy())
        # mermaid_params.write_JSON(lung_reg_params["mermaid_setting_file"])

        del opt
        torch.cuda.empty_cache()

    ###############################
    # Save the results
    np.save("./data/input_noMask_large.npy", I0.detach().cpu().numpy())
    np.save("./data/disp_noMask_large.npy", opt.get_map().detach().cpu().numpy())
    np.save("./data/warped_noMask_large.npy", opt.get_warped_image().detach().cpu().numpy())
    params.write_JSON(path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)