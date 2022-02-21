import argparse
import os
import warnings

import mermaid
import mermaid.module_parameters as pars
import numpy as np
import torch
from mermaid import multiscale_optimizer as MO
from mermaid import utils as MUtils
from tools.evaluate_dir_lab import eval_with_file

warnings.filterwarnings("ignore")

#############################
# Load Params
# path = "./lung_registration_setting.json"
parser = argparse.ArgumentParser(description='3D/2D registration')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')    
parser.add_argument('--result','-r', default='./result')
parser.add_argument('--preprocess','-p',default="")                

def main(args):
    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)

    exp_path = args.disp_f
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)

    torch.autograd.set_detect_anomaly(True)
    #############################
    # Data Preprocessing
    spacing = lung_reg_params["spacing"]
    if args.preprocess == "":
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    else:
        preprocessed_folder = args.preprocess
    
      
    prefix = lung_reg_params["source_img"].split("/")[-3]

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
        mermaid_params_affine['model']['registration_model']['similarity_measure']['type'] = 'ncc'
        mermaid_params_affine['optimizer']["name"]="sgd"
        mermaid_params_affine['optimizer']['sgd']['individual']['lr'] = 1e-03

        affine_opt = MO.SimpleMultiScaleRegistration(I0,
                                                    I1,
                                                    mermaid_spacing,
                                                    sz,
                                                    mermaid_params_affine,
                                                    compute_inverse_map=True)

        affine_opt.get_optimizer().set_visualization(False)
        # affine_opt.get_optimizer().set_visualize_step(50)
        # affine_opt.get_optimizer().set_save_fig(True)
        # affine_opt.get_optimizer().set_expr_name("3d_3d")
        # affine_opt.get_optimizer().set_save_fig_path("./log")
        # affine_opt.get_optimizer().set_pair_name(["affine"])

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

        eval_with_file(lung_reg_params["eval_marker_source_file"],
                    lung_reg_params["eval_marker_target_file"],
                    os.path.join(exp_path, prefix + "_affine_inverse_disp.npy"),
                    croped_dim.copy(),
                    np.flip(np.array(spacing)).copy(),
                    origin,
                    False)

        del affine_opt
        torch.cuda.empty_cache()

    # Performing the lddmm registration
    if lung_reg_params["deformable"]["run"]:

        mermaid_params_proj = pars.ParameterDict()
        mermaid_params_proj.load_JSON(lung_reg_params["deformable"]["setting"])

        try:
          if lung_reg_params["deformable"]["use_affine"]:
            disp_map = torch.from_numpy(np.load(os.path.join(exp_path, prefix + "_affine_disp.npy"))).to(device)
            inverse_map = torch.from_numpy(np.load(os.path.join(exp_path, prefix + "_affine_inverse_disp.npy"))).to(device)
        except:
          disp_map = None
          inverse_map = None
          print("Did not find affine disp map.")

        mermaid_params_proj['optimizer']['single_scale']['nr_of_iterations'] = 5
        mermaid_params_proj['model']['registration_model']['type'] = "svf_vector_momentum_map"#"lddmm_shooting_map"
        mermaid_params_proj['model']['registration_model']['similarity_measure']['sigma'] = 0.05
        mermaid_params_proj['model']['registration_model']['similarity_measure']['type'] = 'lncc'
        # mermaid_params_proj['optimizer']['name'] = 'sgd'
        mermaid_params_proj['optimizer']['sgd']['individual']['lr'] = 0.1
        opt = MO.SimpleMultiScaleRegistration(I0,
                                            I1,
                                            mermaid_spacing,
                                            sz,
                                            mermaid_params_proj,
                                            compute_inverse_map=True)
        if lung_reg_params['deformable']['use_affine'] and (disp_map is not None) and (inverse_map is not None):
          opt.get_optimizer().set_initial_map(disp_map, map0_inverse = inverse_map)

        opt.get_optimizer().set_visualization(False)
        # opt.get_optimizer().set_visualize_step(10)
        # opt.get_optimizer().set_save_fig(True)
        # opt.get_optimizer().set_expr_name("3d_3d")
        # opt.get_optimizer().set_save_fig_path("./log")
        # opt.get_optimizer().set_pair_name(["lddmm"])

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
