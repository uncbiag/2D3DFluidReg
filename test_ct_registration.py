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
import argparse

import warnings
warnings.filterwarnings("ignore")

# Synthesize projection angel
poses = np.array([#[-1., 2., -1.],
                #   [-0.6, 2., -0.8],
                  [-0.3, 3., -0.5],
                #   [-0.2, 2., -0.2],
                  [-0.1, 3., -0.1],
                  # [0., 2., 0.],
                  [0.1, 3., 0.1],
                  #[0.2, 2., 0.2],
                  [0.3, 3., 0.5]])
                #   [0.6, 2., 0.8]])
                  #[1., 2., 1.]])


#############################
# Load Params
# path = "./lung_registration_setting.json"
parser = argparse.ArgumentParser(description='3D/2D registration')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')                    

def main(args):
    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)

    exp_path = args.disp_f
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)

    torch.autograd.set_detect_anomaly(True)
    #############################
    # Data Preprocessing
    resolution_scale = 1.5
    new_spacing = [1.5, 1.5, 1.5]
    sample_rate = [int(1), int(1), int(1)]

    shape = lung_reg_params["shape"]
    spacing = lung_reg_params["spacing"]
    preprocessed_folder = lung_reg_params["preprocessed_folder"]
    prefix = lung_reg_params["source_img"].split("/")[-3]
    if (lung_reg_params["recompute_preprocessed"]):
        preprocessData(lung_reg_params["source_img"],
                      lung_reg_params["target_img"],
                      preprocessed_folder,
                      prefix,
                      shape,
                      spacing,
                      new_spacing,
                      smooth=True,
                      calc_projection=True,
                      poses=poses,
                      resolution_scale=resolution_scale,
                      sample_rate=sample_rate,
                      show_projection=True)


    #############################
    # Load CT data
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Source data
    I0_numpy = np.load(preprocessed_folder+'/' + prefix + '_I0_3d.npy')
    I0 = torch.from_numpy(I0_numpy).unsqueeze(0).unsqueeze(0).to(device)

    # Target data
    I1_numpy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d.npy')
    I1 = torch.from_numpy(I1_numpy).unsqueeze(0).unsqueeze(0).to(device)

    I1_proj = torch.from_numpy(np.load(preprocessed_folder+'/' + prefix + '_I1_proj.npy')).unsqueeze(0).to(device)

    prop = np.load(preprocessed_folder + "/" + prefix + "_prop.npy", allow_pickle = True)
    origin = np.flip(prop.item().get("crop")[0:3])
    croped_dim = np.flip(prop.item().get("dim"))


    #################################
    # Performing the affine registration
    sz = np.array(I0.shape).astype(np.int16)
    mermaid_spacing = 1./(sz[2:] - 1)

    if lung_reg_params["affine"]["run"]:

        mermaid_params_affine = pars.ParameterDict()
        mermaid_params_affine.load_JSON(lung_reg_params["affine"]["setting"])
        mermaid_params_affine['model']['registration_model']['emitter_pos_list'] = poses.tolist()

        affine_opt = MO.SimpleMultiScaleRegistration(I0,
                                                    I1_proj,
                                                    mermaid_spacing,
                                                    sz,
                                                    mermaid_params_affine,
                                                    compute_inverse_map=True)

        affine_opt.get_optimizer().set_model(mermaid_params_affine['model']['registration_model']['type'])
        affine_opt.get_optimizer().add_similarity_measure("projection", Similarity_measure.SdtCTProjectionSimilarity)

        affine_opt.get_optimizer().set_visualization(False)
        affine_opt.get_optimizer().set_visualize_step(50)
        affine_opt.get_optimizer().set_save_fig(True)
        affine_opt.get_optimizer().set_expr_name("3d_2d")
        affine_opt.get_optimizer().set_save_fig_path("./log")
        affine_opt.get_optimizer().set_pair_name(["affine"])

        affine_opt.register()

        disp_map = affine_opt.get_map().detach()

        np.save(os.path.join(exp_path, prefix + "_affine_disp.npy"), disp_map.cpu().numpy())
        np.save(os.path.join(exp_path, prefix + "_affine_warped.npy"), affine_opt.get_warped_image().detach().cpu().numpy())
        inverse_map = MUtils.apply_affine_transform_to_map_multiNC(MUtils.get_inverse_affine_param(affine_opt.get_optimizer().model.Ab),
                                                                affine_opt.get_initial_inverse_map()).detach()
        # print(affine_opt.get_optimizer().model.Ab)
        np.save(os.path.join(exp_path, prefix + "_affine_inverse_disp.npy"), inverse_map.cpu().numpy())

        eval_with_file(lung_reg_params["eval_marker_source_file"],
                    lung_reg_params["eval_marker_target_file"],
                    os.path.join(exp_path, prefix + "_affine_inverse_disp.npy"),
                    croped_dim.copy(),
                    np.flip(np.array(spacing)).copy(),
                    origin,
                    False)


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

        mermaid_params_proj['model']['registration_model']['emitter_pos_list'] = poses.tolist()
        mermaid_params_proj['optimizer']['single_scale']['nr_of_iterations'] = 10
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
        if lung_reg_params['deformable']['use_affine'] and (disp_map is not None) and (inverse_map is not None):
          opt.get_optimizer().set_initial_map(disp_map, map0_inverse = inverse_map)

        opt.get_optimizer().set_visualization(False)
        opt.get_optimizer().set_visualize_step(5)
        opt.get_optimizer().set_save_fig(True)
        opt.get_optimizer().set_expr_name("3d_2d")
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

        eval_with_file(lung_reg_params["eval_marker_source_file"],
                    lung_reg_params["eval_marker_target_file"],
                    os.path.join(exp_path, prefix + "_lddmm_inverse_disp.npy"),
                    croped_dim.copy(),
                    np.flip(np.array(spacing)).copy(),
                    origin,
                    False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)