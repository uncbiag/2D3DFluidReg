import argparse
import os
import random

import matplotlib.pyplot as plt
import mermaid.module_parameters as pars
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.preprocessing import calculate_projection
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from utils.medical_image_utils import binary_dilation, resample, smoother
from utils.plot_helper import show_image_with_colorbar
from utils.pytorch_utils import setup_device

from reconstruction_models.reconstruct_model import recon_model
# from reconstruction_models.reconstruct_model_aug_lag import reconstruct_unconstrained
from reconstruction_models.reconDataset import dirlabDataset

parser = argparse.ArgumentParser(description='3D/2D reconstruction')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')
parser.add_argument('--data', default='dirlab')
parser.add_argument('--result','-r', default='./result')
parser.add_argument('--preprocess','-p',default="")
parser.add_argument('--data_id_path',default="")
parser.add_argument('--exp','-e',default="exp")
parser.add_argument('--with_prior',type=int,default=0)
parser.add_argument('--with_orth_proj',type=int,default=0)
parser.add_argument('--angle', type=int, default=11,
                    help='The scanning range of angles.') 
parser.add_argument('--projection_num', type=int, default=4,
                    help='The number of projection used.')   


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
args = parser.parse_args()

if args.data == "dirlab":
    #Prepare poses 
    angle = 11
    emitter_count = 4
    poses = np.array([
                      [-0.3, 3., -0.2],
                      [-0.1, 3., -0.1],
                      [0.1, 3., 0.1],
                      [0.3, 3., 0.2]])
    ############################
    if args.angle != 0 and args.projection_num != 0:
      angle = args.angle/2.
      emitter_count = args.projection_num
      poses = np.ndarray((emitter_count,3),dtype=np.float)
      poses[:,1] = 3.
      poses[:,0] = np.tan(np.linspace(-angle,angle,num=emitter_count)/180.*np.pi)*3.
      poses[:,2] = np.linspace(-0.2,0.2, num = emitter_count)

    resolution_scale = 1.4
    new_spacing = [1., 1., 1.]
    sample_rate = [int(1), int(1), int(1)]

    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)

    # Prepare data folder path and prefix
    if args.preprocess == "":
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    else:
        preprocessed_folder = args.preprocess
    prefix = lung_reg_params["source_img"].split("/")[-3]

    # Setup the log path and result path
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    log_path = args.result + "/log"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # log_path = "./log/reconstruct_"+ args.data + "/" +args.exp +"/" + prefix
    result_path = args.result+'/'+ prefix + ".npy"

    # Load required data
    if args.with_prior == 1:
        I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I0_3d.npy')
        I_source = torch.from_numpy(I_numpy).unsqueeze(0).unsqueeze(0).to(device)
    else: 
        I_source = None
    I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d.npy')
    I_target = torch.from_numpy(I_numpy).unsqueeze(0).unsqueeze(0).to(device)

    pose_scale = poses
    poses = poses*I_numpy.shape[1]

    I_proj = torch.from_numpy(np.load(preprocessed_folder+'/' + prefix + '_I1_proj.npy')).unsqueeze(1).to(device)

    # If we have warped image, add sim term between warped image and reconstructed image
    if args.disp_f is not "":
        I_warp_npy = np.load(args.disp_f+'/' + prefix + '_lddmm_warped.npy')
        I_seg_npy = np.load(preprocessed_folder+'/' + prefix + '_I1_3d_seg.npy')

        I_seg = torch.from_numpy(I_seg_npy).to(device).unsqueeze(0).unsqueeze(0)
        I_warp = torch.from_numpy(I_warp_npy).to(device)

        # TODO: remove after test.
        I_numpy = np.load(preprocessed_folder+'/' + prefix + '_I0_3d.npy')
        I_source = torch.from_numpy(I_numpy).unsqueeze(0).unsqueeze(0).to(device)
        # I_warp = None
        I_seg = None
    else:
        I_seg = None
        I_warp = None

    if args.with_orth_proj == 1:
        I_proj_orth = torch.sum(I_target, dim=4)
    else: 
        I_proj_orth = None

    # Call reconstruct optimization
    reconstruct(I_proj, I_target, poses, np.array(new_spacing), I_source=I_source, epochs=1000, 
                log_step=100, I_seg=I_seg, I_warped=I_warp, I_proj_orth=I_proj_orth, log_path=log_path, result_path=result_path, pose_scale = pose_scale)

    # reconstruct_unconstrained(I_proj, I_target, poses, np.array(new_spacing), I_source=I_source, 
    #                           log_step=100, I_seg=I_seg, I_warped=I_warp, I_proj_orth=I_proj_orth, log_path=log_path, result_path=result_path, pose_scale = pose_scale)
elif args.data == "sDCT":
    projection_pos_file_path = '../../Data/Raw/NoduleStudyProjections/001/projectionPos.csv'
    projection_dicom_path = '../../Data/Raw/NoduleStudyProjections/001/DICOM'
    sDCT_dicom_path = "../../Data/Raw/DICOMforMN/S00002/SER00001"
    poses_origin = pd.read_csv(projection_pos_file_path).to_numpy()
    poses = np.zeros(poses_origin.shape, dtype=np.float32)
    poses[:,0] = poses_origin[:,1]
    poses[:,1] = poses_origin[:,2]
    poses[:,2] = poses_origin[:,0]
    current_spacing = np.array([0.139, 4, 0.139])
    new_spacing = np.array([1, 4, 1])
    scale_factor = current_spacing/new_spacing
    poses = poses/new_spacing

    # Load projections
    proj_file_list = os.listdir(projection_dicom_path)
    proj_file_list.sort()
    image = [dicom.read_file(projection_dicom_path + '/' + s).pixel_array for s in proj_file_list]
    image = np.array(image)
    image = image.astype(np.float32)/65535+0.0001
    I_proj = torch.from_numpy(image).unsqueeze(1).to(device)
    # # I_proj = F.interpolate(I_proj, scale_factor=scale_factor[0::2])
    # # I_proj = I_proj/(torch.max(I_proj)-torch.min(I_proj))
    # # I_proj[I_proj==0] = 0.1
    I_proj = - torch.log(F.interpolate(I_proj, scale_factor=scale_factor[0::2]))

    # plt.imshow(image[int(image.shape[0]/2),:,:],cmap='gray')
    # plt.axis("off")
    # plt.savefig("./figure/projection.jpg",dpi=300, bbox_inches="tight", pad_inches=0)

    # Load sDCT
    sdct_file_list = os.listdir(sDCT_dicom_path)
    sdct_file_list.sort()
    image  = [dicom.read_file(sDCT_dicom_path + '/' + s).pixel_array for s in sdct_file_list]
    image = np.array(image)
    image = np.transpose(image, (1,0,2))
    image = image.astype(np.float32)/100000.
    I_target = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    I_target = F.interpolate(I_target, scale_factor=scale_factor)

    prefix = projection_dicom_path.split("/")[-2]
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    log_path = "./log/reconstruct_"+ args.data + "/" +prefix
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # reconstruct(I_proj, I_target, poses, new_spacing, 1000, 100, log_path, args.result+'/' + prefix + "_rec.npy")
    # np.save(args.result+'/' + prefix + "_origin.npy", I_target[0,0].cpu().numpy())
elif args.data == 'dirlab_batch':
    #Prepare poses 
    angle = 11
    emitter_count = 4
    poses = np.array([
                      [-0.3, 3., -0.2],
                      [-0.1, 3., -0.1],
                      [0.1, 3., 0.1],
                      [0.3, 3., 0.2]])
    ############################
    if args.angle != 0 and args.projection_num != 0:
      angle = args.angle/2.
      emitter_count = args.projection_num
      poses = np.ndarray((emitter_count, 3), dtype=np.float)
      poses[:, 1] = 3.
      poses[:, 0] = np.tan(np.linspace(-angle, angle, num=emitter_count)/180.*np.pi)*3.
      poses[:, 2] = np.linspace(-0.2, 0.2, num=emitter_count)
    pose_scale = poses
    poses = poses*160.

    resolution_scale = 1.4
    spacing = [2.2, 2.2, 2.2]
    sample_rate = [int(1), int(1), int(1)]

    # Prepare data folder path and prefix
    preprocessed_folder = args.preprocess

    # Setup the log path and result path
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    log_path = args.result + "/log"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    result_path = args.result + "/result"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    setup_device(0)
    # Load required data
    dataset = dirlabDataset(data_path=preprocessed_folder, data_id_path=args.data_id_path, max_num_for_loading=10)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = recon_model(poses, spacing, epochs=4000, log_step=1000,
                           log_path=log_path, result_path=result_path)

    for i, data in enumerate(dataloader):
        I_proj = data[1].to(device)
        I_target = data[0].unsqueeze(0).to(device)
        id = data[2]

        print("###########Start reconstruction for "+dataset.id_list[id])
        # Call reconstruct optimization
        model.run(I_proj, I_target, id=dataset.id_list[id])

