import numpy as np
import matplotlib.pyplot as plt
from test_projection import project_grid
from CTPlayground import resample, win_scale
import torch
import torch.nn.functional as F
import mermaid.example_generation as EG
import mermaid.module_parameters as pars
import Similarity_measure

import mermaid
from mermaid import multiscale_optimizer as MO
import mermaid.module_parameters as pars

# Synthesize projection angel
poses = np.array([[-0.6, 2., -0.06],
                    [-0.5, 2., -0.05],
                    [-0.3, 2., 0.04],
                    [-0.2, 2., 0.03],
                    [-0.1, 2., 0.02],
                    [0., 2., 0.],
                    [0.1, 2., 0.02],
                    [0.2, 2., 0.03],
                    [0.3, 2., 0.04],
                    [0.5, 2., 0.05],
                    [0.6, 2., 0.06]])


def preprocessData(resolution_scale, show_projection, poses):
    new_spacing = [4., 4., 4.]

    shape = (121, 512, 512)
    spacing = [2.5, 0.625, 0.625]
    dtype = np.dtype("<i2")
    fid = open("../eval_data/copd1/copd1/copd1_iBHCT.img", 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    image, spacing = resample(image, np.array(spacing), new_spacing)
    image = image.astype(np.float32)
    [D,W,H] = image.shape
    v = image[10, int(W/2), int(H/2)]
    image[image < 0] = np.mean(image)
    # image = (-image/1000.0+1)*1.673
    np.save("../eval_data/preprocessed/ihale.npy", image)

    I0 = torch.from_numpy(image)

    fid = open("../eval_data/copd1/copd1/copd1_eBHCT.img", 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    image = image.astype(np.float32)
    spacing = [2.5, 0.625, 0.625]
    image, spacing = resample(image, np.array(spacing), new_spacing)
    [D,W,H] = image.shape
    v = image[10, int(W/2), int(H/2)]
    image[image < 0] = np.mean(image)

    # image = (-image/1000.0+1)*1.673

    poses = poses*I0.shape
    spacing = [1., 1., 1.]
    device = torch.device("cuda")
    I1 = torch.from_numpy(image).to(device)
    I1 = I1.unsqueeze(0).unsqueeze(0)
    resolution = [int(I1.shape[2]*resolution_scale), int(I1.shape[3]*resolution_scale)]
    sample_rate = [2,2]
    projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[1])).to(device)
    for i in range(poses.shape[0]):
        grid = torch.flip(project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing),[3])
        projections[0, i] = (torch.sum(F.grid_sample(I1, grid.unsqueeze(0)), dim=4)[0, 0])
        del grid
        torch.cuda.empty_cache()

    np.save("../eval_data/preprocessed/ehale_proj.npy", projections[0].detach().cpu().numpy())
    np.save("../eval_data/preprocessed/ehale_3d.npy", I1[0, 0].detach().cpu().numpy())

    if show_projection:
        I0 = I0.to(device)
        I0 = I0.unsqueeze(0).unsqueeze(0)
        projections_I0 = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[1])).to(device)
        for i in range(poses.shape[0]):
            grid = torch.flip(project_grid(I0, poses[i], (resolution[0], resolution[1]), sample_rate, I0.shape[2:], spacing),[3])
            projections_I0[0, i] = (torch.sum(F.grid_sample(I0, grid.unsqueeze(0)), dim=4)[0, 0])
            del grid
            torch.cuda.empty_cache()

        np.save("../eval_data/preprocessed/ihale_proj.npy", projections_I0[0].detach().cpu().numpy())

        fig, ax = plt.subplots(2, 5)
        for i in range(0, 5):
            ax[0, i].imshow(projections[0, i].detach().cpu().numpy())
        for i in range(0, 5):
            ax[1, i].imshow(projections_I0[0, i].detach().cpu().numpy())
        plt.savefig("./data/projections.png")
    

def project_diagonal(img, delta_theta_x, resolution):
    device = img.device
    theta_x = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).to(device)*torch.cos(delta_theta_x)+\
            torch.Tensor([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]).to(device)*torch.sin(delta_theta_x)

    aff_grid = F.affine_grid(theta_x[0:3, :].unsqueeze(0), resolution)
    return torch.sum(F.grid_sample(img, aff_grid, padding_mode="zeros"), 4)[0,0]

def preprocessDemoData(resolution_scale, show_projection):
    params = pars.ParameterDict()
    params['square_example_images']['len_s'] = 25
    params['square_example_images']['len_l'] = 25
    I0, I1, spacing = EG.CreateSquares(dim=3, add_noise_to_bg=False).create_image_pair(np.array([64, 64, 64]), params=params)

    # Scale down along z direction
    # I1[:, :, :, 0:32, :] = 0

    # I0[:, :, :, :, 30:34] = 0

    I0[:, :, 20:40, 20:35, 15:30] = 0
    I1[:, :, 20:40, 20:40, 15:30] = 0

    I0[:, :, 20:40, 20:35, 34:50] = 0
    I1[:, :, 20:40, 20:45, 34:50] = 0

    np.save("../eval_data/preprocessed/ihale.npy", I0[0,0])

    device = torch.device("cpu")
    I1 = torch.from_numpy(I1).to(device)
    
    resolution = [int(I1.shape[2]*resolution_scale), int(I1.shape[3]*resolution_scale)]
    sample_rate = [2,2]
    projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[1])).to(device)
    for i in range(poses.shape[0]):
        grid = torch.flip(project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing),[3])
        projections[0, i] = (torch.sum(F.grid_sample(I1, grid.unsqueeze(0)), dim=4)[0, 0])
        del grid
        torch.cuda.empty_cache()

    np.save("../eval_data/preprocessed/ehale.npy", projections[0].detach().cpu().numpy())
    np.save("../eval_data/preprocessed/ehale_3d.npy", I1[0, 0].detach().cpu().numpy())

    if show_projection:
        I0 = torch.from_numpy(I0)
        I0 = I0.to(device)
        projections_I0 = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[1])).to(device)
        for i in range(poses.shape[0]):
            grid = torch.flip(project_grid(I0, poses[i], (resolution[0], resolution[1]), sample_rate, I0.shape[2:], spacing),[3])
            projections_I0[0, i] = (torch.sum(F.grid_sample(I0, grid.unsqueeze(0)), dim=4)[0, 0])
            del grid
            torch.cuda.empty_cache()

        fig, ax = plt.subplots(2, 5)
        for i in range(0, 5):
            ax[0, i].imshow(projections[0, i].detach().cpu().numpy())
        for i in range(0, 5):
            ax[1, i].imshow(projections_I0[0, i].detach().cpu().numpy())
        
        # plt.show()
        plt.savefig("./data/projections.png")

def preprocessDemoDataParallel(resolution_scale, theta_list, show_projection):
    params = pars.ParameterDict()
    params['square_example_images']['len_s'] = 20
    params['square_example_images']['len_l'] = 20
    I0, I1, spacing = EG.CreateSquares(dim=3, add_noise_to_bg=False).create_image_pair(np.array([64, 64, 64]), params=params)

    # Scale down along z direction
    # I1[:, :, :, 0:32, :] = 0

    I0[:, :, 20:40, 20:30, 24:40] = 0
    I1[:, :, 20:40, 20:40, 24:40] = 0

    np.save("../eval_data/preprocessed/ihale.npy", I0[0,0])

    device = torch.device("cpu")
    I1 = torch.from_numpy(I1).to(device)
    
    resolution = torch.Size([1,1,int(I1.shape[2]*resolution_scale), int(I1.shape[2]*resolution_scale), int(I1.shape[2]*resolution_scale)])
    projections = torch.zeros((1, theta_list.shape[0], resolution[2], resolution[3])).to(device)
    for i in range(theta_list.shape[0]):
        projections[0, i] = project_diagonal(I1, theta_list[i], resolution)

    np.save("../eval_data/preprocessed/ehale.npy", projections[0].detach().cpu().numpy())
    np.save("../eval_data/preprocessed/ehale_3d.npy", I1[0, 0].detach().cpu().numpy())

    if show_projection:
        I0 = torch.from_numpy(I0)
        I0 = I0.to(device)
        projections_I0 = torch.zeros((1, theta_list.shape[0], resolution[2], resolution[3])).to(device)
        for i in range(theta_list.shape[0]):
            projections_I0[0, i] = project_diagonal(I0, theta_list[i], resolution)

        fig, ax = plt.subplots(2, 5)
        for i in range(0, 5):
            ax[0, i].imshow(projections[0, i].detach().cpu().numpy())
        for i in range(0, 5):
            ax[1, i].imshow(projections_I0[0, i].detach().cpu().numpy())
        
        # plt.show()
        plt.savefig("./data/projections.png")


resolution_scale = 2.0
preprocessData(resolution_scale, True, poses)
# preprocessDemoData(resolution_scale, True)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#############################
# Load Params
path = "./step_by_step_basic_settings_one_opt.json"
params = pars.ParameterDict()
params.load_JSON(path)

#############################
# Load CT data
preprocessed_file_folder = '../eval_data/preprocessed/'
case_pixels = np.load(preprocessed_file_folder+'/ihale.npy')
I0 = torch.from_numpy(case_pixels).unsqueeze(0).unsqueeze(0)
I0 = I0.to(device)


###############################

# case = np.load(preprocessed_file_folder+'/ehale_3d.npy').astype(np.float32)
# case_torch = torch.from_numpy(case).unsqueeze(0).unsqueeze(0)
# I1 = case_torch.to(device)

case = np.load(preprocessed_file_folder+'/ehale_proj.npy').astype(np.float32)
I1 = torch.from_numpy(case).unsqueeze(0).to(device)

###############################
# Set params
params['model']['registration_model']['emitter_pos_list'] = poses.tolist()
params['model']['registration_model']['projection_resolution'] = [int(I0.shape[2]*resolution_scale), int(I0.shape[3]*resolution_scale)]
params['model']['registration_model']['sample_rate'] = [int(2), int(2)]
params['optimizer']['single_scale']['nr_of_iterations'] = 400

params['optimizer']['name'] = 'lbfgs_ls'
# params['optimizer']['sgd']['individual']['lr'] = 0.01
params['optimizer']['scheduler']['patience'] = 3
# params['model']['registration_model']['similarity_measure']['type'] = 'ncc'



#################################
# Performing the registration
sz = np.array(I0.shape)
spacing = 1.0/sz[2:]


# params['model']['registration_model']['type'] = "affine_map"
# params['model']['registration_model']['similarity_measure']['sigma'] = 1.
# affine_opt = MO.SimpleSingleScaleRegistration(I0,
#                                             I1,
#                                             spacing,
#                                             sz,
#                                             params,
#                                             compute_inverse_map=False)
# affine_opt.get_optimizer().set_visualization(True)
# affine_opt.get_optimizer().set_visualize_step(5)
# affine_opt.register()
# I0 = affine_opt.get_warped_image().detach()

# disp_map = torch.from_numpy(np.load("./data/disp_svf.npy")).to(device)

params['model']['registration_model']['type'] = "svf_vector_momentum_map" # "svf_vector_momentum_map"#"lddmm_shooting_map"
# params['model']['registration_model']['similarity_measure']['sigma'] = 0.01
params['model']['registration_model']['similarity_measure']['sigma'] = 0.1
opt = MO.SimpleSingleScaleRegistration(I0,
                                       I1,
                                       spacing,
                                       sz,
                                       params,
                                       compute_inverse_map=False)

opt.get_optimizer().set_model(params['model']['registration_model']['type'])
opt.get_optimizer().add_similarity_measure("projection", Similarity_measure.SdtCTProjectionSimilarity)
# opt.get_optimizer().set_initial_map(disp_map)

opt.get_optimizer().set_visualization(True)
opt.get_optimizer().set_visualize_step(50)
opt.register()

opt.get_optimizer().save_checkpoint("./data/checkpoint")

###############################
# Save the results
np.save("./data/input_svf_one_opt_m_angle.npy", I0.detach().cpu().numpy())
np.save("./data/disp_svf_one_opt_m_angle.npy", opt.get_map().detach().cpu().numpy())
np.save("./data/warped_svf_one_opt_m_angle.npy", opt.get_warped_image().detach().cpu().numpy())
params.write_JSON(path)


