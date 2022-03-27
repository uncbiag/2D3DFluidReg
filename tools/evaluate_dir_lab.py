import argparse
import os

import matplotlib.pyplot as plt
import mermaid.module_parameters as pars
import mermaid.utils as utils
import numpy as np
import torch
import torch.nn.functional as F
from utils.medical_image_utils import resample
from utils.metrics import get_multi_metric
from utils.pytorch_utils import Bilinear


parser = argparse.ArgumentParser(description='Show registration result')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')
parser.add_argument('--preprocess', '-p', metavar='PREPROCESS', default='',
                    help='Path of the folder contains preprocess files.') 

def readPoint(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy list of positions.
    """
    with open(f_path) as fp:
        content = fp.read().split('\n')

        # Read number of points from second
        count = len(content)-1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float32)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split('\t')
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])

        return points


def calc_warped_points(source_list_t, phi_t, dim, spacing):
    """
    :param source_list_t: source image.
    :param phi_t: the inversed displacement.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :return: a N*3 tensor containg warped positions in the physical coordinate.
    """
    warped_list_t = F.grid_sample(phi_t, source_list_t, align_corners=True)

    warped_list_t = torch.flip(warped_list_t.permute(0, 2, 3, 4, 1), [4])[0, 0, 0]
    # warped_list_t = warped_list_t.permute(0, 2, 3, 4, 1)[0, 0, 0]
    warped_list_t = torch.mul(torch.mul(warped_list_t, torch.from_numpy(dim-1.)), torch.from_numpy(spacing))

    return warped_list_t

def eval_with_file(source_file, target_file, phi_file, dim, spacing, origin, plot_result=False):
    """
    :param source_file: the path to the position of markers in source image.
    :param target_file: the path to the position of markers in target image.
    :param phi_file: the path to the displacement map.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :param plot_result: a bool value indicating whether to plot the result.
    """
    source_list = readPoint(source_file)
    target_list = readPoint(target_file)
    phi = np.load(phi_file)

    res, res_seperate = eval_with_data(
        source_list, target_list, phi, dim, spacing, origin, plot_result)
    return res, res_seperate

def eval_with_data(source_list, target_list, phi, dim, spacing, origin, plot_result=False):
    """
    :param source_list: a numpy list of markers' position in source image.
    :param target_list: a numpy list of markers' position in target image.
    :param phi: displacement map in numpy format.
    :param dim: voxel dimenstions.
    :param spacing: image spacing.
    :param return: res, [dist_x, dist_y, dist_z] res is the distance between 
    the warped points and target points in MM. [dist_x, dist_y, dist_z] are 
    distances in MM along x,y,z axis perspectively.
    """
    origin_list = np.repeat([origin,], target_list.shape[0], axis=0)

    target_list_t = torch.from_numpy((target_list-1.)*spacing)

    source_list_norm = (source_list-1.-origin_list)/(dim-1.)*2.0-1.0
    source_list_t = torch.from_numpy(
        source_list_norm).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    phi_t = torch.from_numpy(phi).double()

    warped_list_t = calc_warped_points(source_list_t, phi_t, dim, spacing)
    # np.save( "./log/marker_warped.npy", warped_list_t.numpy())
    warped_list_t = warped_list_t + torch.from_numpy((origin_list)*spacing)
    # np.save( "./log/marker_warped_target_coord.npy", warped_list_t.numpy())

    pdist = torch.nn.PairwiseDistance(p=2)
    dist = pdist(target_list_t, warped_list_t)
    idx = torch.argsort(dist).numpy()
    # np.save("./log/marker_most_inaccurate.npy", idx)
    dist_x = torch.mean(torch.abs(target_list_t[:,0] - warped_list_t[:,0]))
    dist_y = torch.mean(torch.abs(target_list_t[:,1] - warped_list_t[:,1]))
    dist_z = torch.mean(torch.abs(target_list_t[:,2] - warped_list_t[:,2]))
    res = torch.mean(dist)
    print("The mean distance is %f and the mean distance along x,y,z are %f, %f, %f"%(res, dist_x, dist_y, dist_z))

    if plot_result:
        source_list_eucl = source_list*spacing
        fig, axes = plt.subplots(3,1)
        for i in range(3):
            axes[i].plot(target_list_t[:100,i].cpu().numpy(), "+", markersize=2, label="source")
            axes[i].plot(warped_list_t[:100,i].cpu().numpy(), '+', markersize=2, label="warped")
            axes[i].plot(source_list_eucl[:100,i], "+", markersize=2, label="target")
            axes[i].set_title("axis = %d"%i)
            
        plt.legend()
        # plt.show()
        plt.savefig("./log/eval_dir_lab_reg.png", bbox_inches="tight", dpi=300)

    return res, [dist_x, dist_y, dist_z]

def plot_on_img(img, x, y, ax):
    ax.imshow(img)
    for i in range(len(x)):
        ax.scatter(x, y, s=1)

def plot_line(img, source_list, target_list, ax, style, text):
    
    for i in range(len(source_list)):
        start = source_list[i]
        end = target_list[i]
        ax.plot([start[0], end[0]], [start[1], end[1]], style, linewidth=1)
    plt.sca(ax)
    plt.title(text)

def plot_arrow(img, start_list, end_list, ax, style):
    ax.imshow(img)
    for i in range(len(start_list)):
        start = start_list[i]
        delta = end_list[i] - start
        ax.arrow(start[0], start[1], delta[0], delta[1], length_includes_head=True, head_width=5, head_length=2, color="r")

def plot_marker_distribution(source_file, target_file, spacing_origin, ct_source_path, ct_target_path, spacing_npy):
    ct_source, new_spacing = resample(np.load(ct_source_path), spacing_npy, [1., 1., 1.])
    ct_target, new_spacing = resample(np.load(ct_target_path), spacing_npy, [1., 1., 1.])
    (D,W,H) = ct_source.shape
    D_mid = int(D/2)
    W_mid = int(W/2)
    H_mid = int(H/2)
    # source_list = readPoint(source_file)*spacing_origin
    source_list = np.load("./log/points.npy")*spacing_origin
    target_file = readPoint(target_file)*spacing_origin

    fig, axes = plt.subplots(2, 3)
    plot_on_img(ct_source[D_mid, :, :], source_list[:, 0], source_list[:, 1], axes[0, 0])
    plot_on_img(ct_source[:, W_mid, :], source_list[:, 0], source_list[:, 2], axes[0, 1])
    plot_on_img(ct_source[:, :, H_mid], source_list[:, 1], source_list[:, 2], axes[0, 2])
    plot_on_img(ct_target[D_mid, :, :], target_file[:, 0], target_file[:, 1], axes[1, 0])
    plot_on_img(ct_target[:, W_mid, :], target_file[:, 0], target_file[:, 2], axes[1, 1])
    plot_on_img(ct_target[:, :, H_mid], target_file[:, 1], target_file[:, 2], axes[1, 2])
    plt.savefig("./log/marker_distribution.png")

def showOverlay(a, b, axes):
    merged = np.zeros((a.shape[0], a.shape[1], 3))
    merged[:,:,0] = (a-np.min(a))/(np.max(a)-np.min(a))
    merged[:,:,1] = (b-np.min(b))/(np.max(b)-np.min(b))
    
    axes.imshow(merged)

def plot_marker_deformation(source_file, target_file, phi_file, dim_origin, spacing_origin, origin, ct_source_path, ct_target_path, warped_file, spacing_npy, label=""):
    ct_source, new_spacing = resample(np.load(ct_source_path), spacing_npy, [1., 1., 1.])
    ct_target, new_spacing = resample(np.load(ct_target_path), spacing_npy, [1., 1., 1.])
    # plot_all(ct_source, label=label + "source")
    # plot_all(ct_target, label=label + "target")

    (D,W,H) = ct_source.shape
    D_mid = int(D/2)
    W_mid = int(W/2)
    H_mid = int(H/2)

    warped_img, new_spacing = resample(np.load(warped_file)[0, 0], spacing_npy, [1., 1., 1.])

    source_list = readPoint(source_file)
    target_list = readPoint(target_file)
    phi = np.load(phi_file)

    # TODO: Show part of the markers
    # source_list = source_list[::3]
    # target_list = target_list[::3]

    origin_list = np.repeat([origin,], target_list.shape[0], axis=0)

    source_list_norm = (source_list-1.-origin_list)/(dim_origin-1.)*2.0-1.0
    source_list_t = torch.from_numpy(source_list_norm).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    phi_t = torch.from_numpy(phi).double()

    warped_list = calc_warped_points(source_list_t, phi_t, dim_origin, spacing_origin).numpy()

    source_list = (source_list-origin_list)*spacing_origin
    target_list = (target_list-origin_list)*spacing_origin

    fig, axes = plt.subplots(3,3)
    plot_line(ct_source[D_mid, :, :], source_list[:, 0:2], target_list[:, 0:2], axes[0, 0], "y-", "source")
    plot_line(ct_source[:, W_mid, :], source_list[:, 0::2], target_list[:, 0::2], axes[0, 1], "y-", "source")
    plot_line(ct_source[:, :, H_mid], source_list[:, 1:], target_list[:, 1:], axes[0, 2], "y-", "source")
    plot_line(ct_source[D_mid, :, :], source_list[:, 0:2], warped_list[:, 0:2], axes[0, 0], "r-", "source")
    plot_line(ct_source[:, W_mid, :], source_list[:, 0::2], warped_list[:, 0::2], axes[0, 1], "r-", "source")
    plot_line(ct_source[:, :, H_mid], source_list[:, 1:], warped_list[:, 1:], axes[0, 2], "r-", "source")
    axes[0,0].imshow(ct_source[D_mid, :, :])
    axes[0,1].imshow(ct_source[:, W_mid, :])
    axes[0,2].imshow(ct_source[:, :, H_mid])

    # warped_img = ct_source

    merged = np.zeros((warped_img.shape[1], warped_img.shape[2], 3))
    warped_img_d = warped_img[D_mid, :, :]
    warped_img_d = (warped_img_d-np.min(warped_img_d))/(np.max(warped_img_d)-np.min(warped_img_d))
    target_d = ct_target[D_mid, :, :]
    target_d = (target_d-np.min(target_d))/(np.max(target_d)-np.min(target_d))
    merged[:,:,0] = warped_img_d
    merged[:,:,1] = target_d
    
    axes[1, 0].imshow(merged)
    plot_line(merged, source_list[:, 0:2], target_list[:, 0:2], axes[1, 0], "y-", "warped")
    plot_line(warped_img[:, W_mid, :], source_list[:, 0::2], target_list[:, 0::2], axes[1, 1], "y-", "warped")
    plot_line(warped_img[:, :, H_mid], source_list[:, 1:], target_list[:, 1:], axes[1, 2], "y-", "warped")
    plot_line(merged, source_list[:, 0:2], warped_list[:, 0:2], axes[1, 0], "r-", "warped")
    plot_line(warped_img[:, W_mid, :], source_list[:, 0::2], warped_list[:, 0::2], axes[1, 1], "r-", "warped")
    plot_line(warped_img[:, :, H_mid], source_list[:, 1:], warped_list[:, 1:], axes[1, 2], "r-", "warped")
    showOverlay(warped_img[D_mid, :, :], ct_source[D_mid, :, :], axes[1,0])
    showOverlay(warped_img[:, W_mid, :], ct_source[:, W_mid, :], axes[1,1])
    showOverlay(warped_img[:, :, H_mid], ct_source[:, :, H_mid], axes[1,2])

    plot_line(ct_target[D_mid, :, :], source_list[:, 0:2], target_list[:, 0:2], axes[2, 0], "y-", "target")
    plot_line(ct_target[:, W_mid, :], source_list[:, 0::2], target_list[:, 0::2], axes[2, 1], "y-", "target")
    plot_line(ct_target[:, :, H_mid], source_list[:, 1:], target_list[:, 1:], axes[2, 2], "y-", "target")
    plot_line(ct_target[D_mid, :, :], source_list[:, 0:2], warped_list[:, 0:2], axes[2, 0], "r-", "target")
    plot_line(ct_target[:, W_mid, :], source_list[:, 0::2], warped_list[:, 0::2], axes[2, 1], "r-", "target")
    plot_line(ct_target[:, :, H_mid], source_list[:, 1:], warped_list[:, 1:], axes[2, 2], "r-", "target")
    axes[2,0].imshow(ct_target[D_mid, :, :])
    axes[2,1].imshow(ct_target[:, W_mid, :])
    axes[2,2].imshow(ct_target[:, :, H_mid])

    plt.show()
    # plt.savefig("./log/marker_deformation_" + label + ".png", dpi=300) 

def plot_all(img, label=""):
    (D, W, H) = img.shape
    D = int(D/5)
    row = int(D/5)
    fig, axes = plt.subplots(row, 5)
    for i in range(row):
        for j in range(5):
            axes[i, j].imshow(img[(i*5 + j)*5])
    plt.savefig("./log/plot_all_" + label + ".png", dpi=300)

def plot_one_marker_per_image(img, marker_pos, ax, rowId):
    marker_depth = int(marker_pos[2])
    start = max(0, marker_depth-5)
    for i in range(10):
        ax[rowId, i].imshow(img[start + i])
    
    ax[rowId, marker_depth-start].plot(marker_pos[0], marker_pos[1], markersize=2, marker="o", color="red")
    ax[rowId, 0].set_xlabel("pos:" + str(start))


def plot_one_marker(source_file, target_file, phi_file, dim_origin, spacing_origin, ct_source_path, ct_target_path, warped_file, spacing_npy, label=""):
    ct_source, new_spacing = resample(np.load(ct_source_path), spacing_npy, [1., 1., 1.])
    ct_target, new_spacing = resample(np.load(ct_target_path), spacing_npy, [1., 1., 1.])
    warped_img, new_spacing = resample(np.load(warped_file)[0, 0], spacing_npy, [1., 1., 1.])

    source_list = readPoint(source_file)
    target_list = readPoint(target_file)
    phi = np.load(phi_file)

    source_list_norm = (source_list-1.)/(dim_origin-1.)*2.0-1.0
    source_list_t = torch.from_numpy(source_list_norm).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    phi_t = torch.from_numpy(phi).double()

    warped_list = calc_warped_points(source_list_t, phi_t, dim_origin, spacing_origin).cpu().numpy()
    source_list = source_list*spacing_origin
    target_list = target_list*spacing_origin

    dist = np.sum(np.power(target_list-warped_list, 2), axis=1)
    index = 298 # np.argmax(dist)

    fig, axes = plt.subplots(3,10)
    plot_one_marker_per_image(ct_source, source_list[index], axes, 0)
    plot_one_marker_per_image(ct_target, target_list[index], axes, 1)
    plot_one_marker_per_image(warped_img, warped_list[index], axes, 2)

    # plt.show()
    plt.savefig("./log/plot_one_marker_" + label + ".png")



if __name__ == "__main__":
    args = parser.parse_args()
    # Load Params
    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)
    disp_folder = args.disp_f
    if args.preprocess == "":
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    else:
        preprocessed_folder = args.preprocess
    
    prefix = lung_reg_params["source_img"].split("/")[-3]

    source_file = lung_reg_params["eval_marker_source_file"]
    target_file = lung_reg_params["eval_marker_target_file"]
    phi_file = disp_folder + "/" + prefix + "_affine_inverse_disp.npy"
    
    prop_file = preprocessed_folder + '/' + prefix +'_prop.npy'
    
    prop = np.load(prop_file, allow_pickle=True)
    dim = np.flip(np.array(prop.item().get("dim")))
    origin = np.flip(prop.item().get("crop")[0:3])
    spacing = np.flip(lung_reg_params["spacing"]).copy()

    eval_with_file(source_file, target_file, phi_file, dim, spacing, origin, False)

    phi_file = disp_folder + "/" + prefix + "_lddmm_inverse_disp.npy"
    eval_with_file(source_file, target_file, phi_file, dim, spacing, origin, False)

    source_seg = torch.from_numpy(np.load(f"{preprocessed_folder}/{prefix}_I0_3d_seg.npy")).float().unsqueeze(0).unsqueeze(0)
    target_seg = torch.from_numpy(np.load(f"{preprocessed_folder}/{prefix}_I1_3d_seg.npy")).float().unsqueeze(0).unsqueeze(0)
    bilinear = Bilinear(zero_boundary=True, using_scale=False, mode="nearest")
    phi = torch.from_numpy(np.load(f"{disp_folder}/{prefix}_lddmm_disp.npy")*2.-1.)
    warped = bilinear(source_seg, phi)

    metrics = get_multi_metric(warped, target_seg, verbose=False)
    print(f"Dice:{metrics['batch_avg_res']['dice'][0, 1:]}")

    # ct_source_file = "../eval_data/preprocessed/ihale_3d.npy"
    # ct_target_file = "../eval_data/preprocessed/ehale_3d.npy"
    # plot_marker_distribution(source_file, target_file, spacing, ct_source_file, ct_target_file, np.array([4., 4., 4.]))

    # ct_source_file = lung_reg_params["preprocessed_folder"] + "/" + prefix +"_I0_3d.npy"
    # ct_target_file = lung_reg_params["preprocessed_folder"] + "/" + prefix +"_I1_3d.npy"
    # warped_file = disp_folder + "/" + prefix + "_lddmm_warped.npy"

    # plot_marker_deformation(source_file, 
    #                         target_file, 
    #                         phi_file, 
    #                         dim, spacing, origin, 
    #                         ct_source_file, 
    #                         ct_target_file, 
    #                         warped_file, 
    #                         np.array([1.5, 1.5, 1.5]),
    #                         "lddmm")
