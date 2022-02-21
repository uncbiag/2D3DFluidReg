
import numpy as np
import torch
import torch.nn.functional as F

from utils.medical_image_utils import resample, seg_bg_mask, load_IMG, smoother
import mermaid.module_parameters as pars

import matplotlib.pyplot as plt
import os 

import argparse
parser = argparse.ArgumentParser(description='3D/2D registration preprocess')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--preprocess', '-p', metavar='PREPROCESS', default='.',
                    help='Path of the folder contains preprocess files.')  
parser.add_argument('--angle', type=int, default=11,
                    help='The scanning range of angles.') 
parser.add_argument('--projection_num', type=int, default=4,
                    help='The number of projection used.')   
parser.add_argument('--resolution_scale', type=float, default=1.4,
                    help='The number of projection used.')   

def showAll(img):
    amount = img.shape[0]
    rowCount = int(amount/10)
    if amount%10 != 0:
        rowCount = rowCount + 1
    fig, axes = plt.subplots(rowCount, 10)
    for i in range(0, rowCount):
        for j in range(0, 10):
            index = i * 10 + j
            if index >= amount: 
                break
            axes[i, j].imshow(img[index])
    plt.show()

def project_grid_multi(emi_pos, resolution, sample_rate, obj_shape, spacing, device, dtype):
        # Axes definition: 0-axial, 1-coronal, 2-sagittal
        # sample_rate: sample count per pixel
        d, w, h = obj_shape
        (res_d, res_h) = resolution
        sr_d, sr_w, sr_h = sample_rate

        # P0 - one point in each coronal plane of CT. We use the points at Y axies.
        # I0 - start point of each rays.
        # N - Normal vector of coronal plane.
        P0 = torch.mm(
            torch.linspace(0, w-1, sr_w*w, device=device, dtype=dtype).unsqueeze(1), 
            torch.tensor([[0., 1., 0.]], device=device, dtype=dtype))
        I0 = torch.from_numpy(emi_pos).to(device).unsqueeze(1).unsqueeze(1).type(dtype)
        N = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
        
        # Calculate direction vectors for each rays
        lin_x = torch.linspace(-res_d/2, res_d/2-1, steps=res_d*sr_d)
        lin_y = torch.linspace(-res_h/2, res_h/2-1, steps=res_h*sr_h)
        grid_x, grid_y = torch.meshgrid(lin_x, lin_y)
        I = torch.zeros((lin_x.shape[0], lin_y.shape[0], 3), device=device, dtype=dtype)
        I[:,:,0] = grid_x
        I[:,:,2] = grid_y
        I = torch.add(I,-I0)
        dx = torch.mul(I, 1./I[:,:,:,1:2])
        I = I/torch.norm(I, dim=3, keepdim=True)
        dx = torch.norm(dx*spacing.to(device).unsqueeze(0).unsqueeze(0), dim=3)

        # Define a line as I(t)=I0+t*I
        # Define a plane as (P-P0)*N=0, P is a vector of points on the plane
        # Thus at the intersection of the line and the plane, we have d=(P0-I0)*N/(I*N)
        # Then we can get the position of the intersection by I(t) = t*I + I0
        # T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(2), torch.matmul(P0-I0, N).unsqueeze(0))
        # grid = torch.add(torch.matmul(T.unsqueeze(3), I.unsqueeze(2)), I0)

        T = torch.matmul(1./(torch.matmul(I,N)).unsqueeze(3).unsqueeze(4), torch.matmul(P0-I0, N).unsqueeze(1).unsqueeze(1))
        grid = torch.add(torch.matmul(I.unsqueeze(4), T).permute(0,1,2,4,3), I0.unsqueeze(1))

        # Since grid_sample function accept input in range (-1,1)
        grid[:,:,:,:,0] = grid[:,:,:,:,0]/obj_shape[0]*2.0
        grid[:,:,:,:,1] = (grid[:,:,:,:,1]-0.)/obj_shape[1]*2.0 + -1.
        grid[:,:,:,:,2] = grid[:,:,:,:,2]/obj_shape[2]*2.0
        return grid, dx


def calculate_projection(img, poses_scale, resolution_scale, sample_rate, spacing, device):
    
    poses = poses_scale*img.shape[1]
    spacing = torch.tensor(spacing).to(device)
    I0 = torch.from_numpy(img).to(device)
    I0 = I0.unsqueeze(0).unsqueeze(0)
    resolution = [int(I0.shape[2] * resolution_scale),
                  int(I0.shape[4] * resolution_scale)]
    grids, dx = project_grid_multi(poses, resolution, sample_rate, I0.shape[2:], spacing, I0.device, I0.dtype)
    grids = torch.flip(grids, [4])
    (p, d, h, w) = grids.shape[0:4]
    b = I0.shape[0]
    grids = torch.reshape(grids, (1,1,1,-1,3))
    # dx = dx.unsqueeze(1).unsqueeze(1)
    I0_proj = torch.mul(torch.sum(F.grid_sample(I0, grids, align_corners = True).reshape((b, p, d, h, w)), dim=4), dx).float()

    # projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
    # for i in range(poses.shape[0]):
    #     grid, dx = project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing)
    #     grid = torch.flip(grid,[3])
    #     dx = dx.unsqueeze(0).unsqueeze(0)
    #     projections[0, i] = torch.mul(torch.sum(F.grid_sample(I1, grid.unsqueeze(0), align_corners=False), dim=4), dx)[0, 0]
    #     # np.save("./log/grids_sim_matrix_"+str(i)+".npy", grid.cpu().numpy())
    #     del grid
    #     torch.cuda.empty_cache()

    proj = I0_proj[0].detach().cpu().numpy()
    del I0_proj, grids, I0, dx, spacing
    torch.cuda.empty_cache()
    return proj


def project_grid(img, emi_pos, resolution, sample_rate, obj_shape, spacing):
    d, w, h = obj_shape
    res_d, res_h = resolution
    device = img.device
    emi_pos_s = emi_pos
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
    dx = torch.mul(I, 1./I[:,:,1:2])
    I = I/torch.norm(I, dim=2, keepdim=True)
    dx = torch.norm(dx*spacing.unsqueeze(0).unsqueeze(0), dim=2)

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
    return grid, dx

# def calculate_projection(img, poses_scale, resolution_scale, sample_rate, spacing, device):
#     poses = poses_scale*img.shape[1]
#     spacing = torch.tensor(spacing).to(device)
#     I1 = torch.from_numpy(img).to(device)
#     I1 = I1.unsqueeze(0).unsqueeze(0)
#     resolution = [int(I1.shape[2] * resolution_scale),
#                   int(I1.shape[4] * resolution_scale)]
#     projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
#     for i in range(poses.shape[0]):
#         grid, dx = project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing)
#         grid = torch.flip(grid,[3])
#         dx = dx.unsqueeze(0).unsqueeze(0)
#         projections[0, i] = torch.mul(torch.sum(F.grid_sample(I1, grid.unsqueeze(0), align_corners=True), dim=4), dx)[0, 0]
#         # np.save("./log/grids_sim_matrix_"+str(i)+".npy", grid.cpu().numpy())
#         del grid
#         torch.cuda.empty_cache()

#     projections_npy = projections[0].detach().cpu().numpy()
#     del projections, dx
#     torch.cuda.empty_cache()
#     return projections_npy

def preprocessData(source_file, target_file, dest_folder, dest_prefix, shape, spacing,
                   new_spacing, smooth=False, sigma=6, calc_projection=False, poses_scale=[],
                   resolution_scale=1.0, sample_rate=[1, 1], show_projection=False):
    print("Preprocessing data...")

    img_0, mask_0, bbox_0 = load_IMG(source_file, shape, spacing, new_spacing)
    img_1, mask_1, bbox_1 = load_IMG(target_file, shape, spacing, new_spacing)

    # Figure out the bbox size
    bbox = np.ndarray((6)).astype(np.int)
    shape = img_0.shape
    for i in range(0,3):
        bbox[i] = max(min(bbox_0[i], bbox_1[i])-20, 0)
        bbox[i+3] = min(max(bbox_0[i+3], bbox_1[i+3])+20, shape[i]) 
    prop = {'crop': bbox}
    prop["dim"] = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]

    # Crop and resample the image
    img_0 = img_0 * mask_0 
    img_0 = img_0.astype(np.float32)
    img_0 = img_0[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    mask_0 = mask_0[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    img_0, _ = resample(img_0, np.array(spacing), new_spacing)
    mask_0, _ = resample(mask_0, np.array(spacing), new_spacing)

    img_1 = img_1 * mask_1 
    img_1 = img_1.astype(np.float32)
    img_1 = img_1[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    mask_1 = mask_1[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    img_1, _ = resample(img_1, np.array(spacing), new_spacing)
    mask_1, _ = resample(mask_1, np.array(spacing), new_spacing)

    # Smooth the image
    if smooth:
        img_0 = smoother(img_0, sigma=sigma)
        img_1 = smoother(img_1, sigma=sigma)

    # Save the 3d image
    np.save(dest_folder + "/" + dest_prefix + "_I0_3d.npy", img_0)
    np.save(dest_folder + "/" + dest_prefix + "_I1_3d.npy", img_1)
    np.save(dest_folder + "/" + dest_prefix + "_prop.npy", prop)

    np.save(dest_folder + "/" + dest_prefix + "_I0_3d_seg.npy", mask_0)
    np.save(dest_folder + "/" + dest_prefix + "_I1_3d_seg.npy", mask_1)

    # Calculate the orthogmal projection for experiments
    # Remove when finishing the experiments
    # img_proj_orth_0 = np.sum(img_0, 2)
    # img_proj_orth_1 = np.sum(img_1, 2)
    # np.save(dest_folder + "/" + dest_prefix + "_I0_proj_orth.npy", img_proj_orth_0)
    # np.save(dest_folder + "/" + dest_prefix + "_I1_proj_orth.npy", img_proj_orth_1)

    # Calculate the projection image
    if calc_projection:
        device = torch.device("cuda")
        img_proj_0 = calculate_projection(img_0, poses_scale, resolution_scale,
                                          sample_rate, new_spacing, device)
        np.save(dest_folder + "/" + dest_prefix + "_I0_proj.npy", img_proj_0)

        img_proj_1 = calculate_projection(img_1, poses_scale, resolution_scale,
                                          sample_rate, new_spacing, device)
        np.save(dest_folder + "/" + dest_prefix + "_I1_proj.npy", img_proj_1)

        if show_projection:
            step = min(1, int(img_proj_0.shape[0]/10))
            fig, ax = plt.subplots(2, img_proj_0.shape[0])
            for i in range(0, img_proj_0.shape[0]):
                im = ax[0, i].imshow(img_proj_0[i])
                cb = plt.colorbar(im, ax = ax[0,i], shrink=0.4, pad=0.01, orientation="horizontal")
                cb.ax.tick_params(labelsize=3, rotation=45)
                im = ax[1, i].imshow(img_proj_1[i])
                cb = plt.colorbar(im, ax = ax[1,i], shrink=0.4, pad=0.01, orientation="horizontal")
                cb.ax.tick_params(labelsize=3, rotation=45)
            ax[0,0].set_ylabel("Source projection")
            ax[1,0].set_ylabel("Target projection")
            plt.savefig("./log/" + dest_prefix + "_projections.png", dpi=300)
            # plt.show()
    
    

if __name__ == "__main__":
    args = parser.parse_args()
    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)

    # Synthesize projection angle
    angle = 5
    emitter_count = 4
    poses_scale = np.array([
                      [-0.3, 3., -0.2],
                      [-0.1, 3., -0.1],
                      [0.1, 3., 0.1],
                      [0.3, 3., 0.2]])
    ############################
    if args.angle != 0 and args.projection_num != 0:
        angle = args.angle/2.
        emitter_count = args.projection_num
        poses_scale = np.ndarray((emitter_count,3),dtype=np.float)
        poses_scale[:,1] = 3.
        poses_scale[:,0] = np.tan(np.linspace(-angle,angle,num=emitter_count)/180.*np.pi)*3.
        poses_scale[:,2] = np.linspace(-0.2,0.2, num = emitter_count)

    torch.autograd.set_detect_anomaly(True)
    #############################
    # Data Preprocessing
    resolution_scale = args.resolution_scale
    new_spacing = [1., 1., 1.]
    sample_rate = [int(1), int(1), int(1)]

    shape = lung_reg_params["shape"]
    spacing = lung_reg_params["spacing"]
    if args.preprocess == "":
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    else:
        preprocessed_folder = args.preprocess

    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder, exist_ok=True)
    
    # prefix = lung_reg_params["source_img"].split("/")[-3] + "_"+str(int(angle*2))+"_degree_"+ str(emitter_count)
    prefix = lung_reg_params["source_img"].split("/")[-3]
    if (lung_reg_params["recompute_preprocessed"]):
        preprocessData(lung_reg_params["source_img"], 
                      lung_reg_params["target_img"],
                      preprocessed_folder,
                      prefix,
                      shape,
                      spacing,
                      new_spacing,
                      smooth=False,
                      sigma = 2,
                      calc_projection=True,
                      poses_scale=poses_scale,
                      resolution_scale=resolution_scale,
                      sample_rate=sample_rate,
                      show_projection=False)
