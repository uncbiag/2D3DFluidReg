from preprocessing import load_IMG
import mermaid.module_parameters as pars
from evaluate_dir_lab import readPoint
from medical_image_utils import resample
import matplotlib.pyplot as plt
import numpy as np
from mermaid import utils
import torch
import argparse
import SimpleITK as sitk

parser = argparse.ArgumentParser(description='Show registration result')
parser.add_argument('--setting', '-s', metavar='SETTING', default='',
                    help='setting')
parser.add_argument('--preprocess', '-p', metavar='PREPROCESS', default='',
                    help='Path of the folder contains preprocess files.') 
parser.add_argument('--disp_f', '-d', metavar='DISP_F', default='',
                    help='Path of the folder contains displacement files.')

def load_IMG(file_path, shape, spacing, new_spacing):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    mask = None
    bbox = None
    # mask, bbox = seg_bg_mask(image, True)

    # for i in range(0,10):
    #     plt.imshow((mask*image)[:,:,i*20])
    #     plt.savefig("./log/image_%i.jpg"%i)
    
    image = image.astype(np.float32)
    # image_max = np.max(image)
    # image_min = np.min(image)
    # image = image/(image_max - image_min)

    # image[image>300] = 0
    # image = (image/1000.0+1)*1.673

    return image, mask, bbox

def nearest_int(x):
    if x - int(x) > 0.5:
        return int(x) + 1
    else:
        return int(x)


def plot_dot(img, x, y, ax, title):
    ax.imshow(img)
    ax.scatter(y, x, s=100, facecolors="none", edgecolors='w', alpha=0.5)
    plt.sca(ax)
    plt.title(title)

def main(args):
    # # Load Params
    lung_reg_params = pars.ParameterDict()
    lung_reg_params.load_JSON(args.setting)

    new_spacing = [1., 1., 1.]
    shape = lung_reg_params["shape"]
    spacing = lung_reg_params["spacing"]
    prefix = lung_reg_params["source_img"].split("/")[-3]

    source_ori, mask, bbox = load_IMG(lung_reg_params["source_img"], shape, spacing, new_spacing)
    source, _ = resample(source_ori, np.array(spacing), new_spacing)
    target_ori, mask, bbox = load_IMG(lung_reg_params["target_img"], shape, spacing, new_spacing)
    target, _ = resample(target_ori, np.array(spacing), new_spacing)

    # warped = np.load(lung_reg_params["projection"]["warped_file"])[0, 0]
    # warped, _ = resample(warped, np.array([1.5, 1.5, 1.5]), new_spacing)

    if args.preprocess != "":
        preprocessed_folder = args.preprocess
    else: 
        preprocessed_folder = lung_reg_params["preprocessed_folder"]
    disp_folder = args.disp_f
    prefix = lung_reg_params["source_img"].split("/")[-3]
    
    prop_file = lung_reg_params["preprocessed_folder"] + "/" + prefix + '_prop.npy'
    prop = np.load(prop_file, allow_pickle=True)
    bbox = prop.item().get("crop")
    croped = source_ori[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    origin = np.flip(prop.item().get("crop")[0:3])*np.flip(spacing)
    # warped, _ = resample(croped, np.array(spacing), new_spacing)


    croped, _ = resample(croped, np.array(spacing), np.array([1.5, 1.5, 1.5]))
    phi_file = disp_folder + "/" + prefix + "_affine_inverse_disp.npy"
    phi = np.load(phi_file)
    warped = utils.compute_warped_image_multiNC(
        torch.from_numpy(croped).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(phi), 1./(np.array(croped.shape)-1),
        1, zero_boundary=True)
    warped, _ = resample(warped[0,0].numpy(), np.array([1.5, 1.5, 1.5]), new_spacing)


    source_marker_file = lung_reg_params["eval_marker_source_file"]
    target_marker_file = lung_reg_params["eval_marker_target_file"]

    marker_list = np.flip(np.load("./log/marker_most_inaccurate.npy"))[0:1]
    source_marker = np.array(readPoint(source_marker_file)-1)*np.flip(spacing)
    target_marker = np.array(readPoint(target_marker_file)-1)*np.flip(spacing)
    warped_marker = np.load("./log/marker_warped.npy")
    warped_marker_in_target = np.load("./log/marker_warped_target_coord.npy")

    for i in range(marker_list.shape[0]):
        m_s = source_marker[marker_list[i]]
        m_t = target_marker[marker_list[i]]
        m_w = warped_marker[marker_list[i]]
        m_w_t = warped_marker_in_target[marker_list[i]]
        fig, axes = plt.subplots(4, 3)
        plot_dot(source[nearest_int(m_s[2]), :, :], m_s[1], m_s[0], axes[0, 0], "z=%i"%nearest_int(m_s[2]))
        axes[0, 0].set_ylabel("marker in source image.")
        plot_dot(source[:, nearest_int(m_s[1]), :], m_s[2], m_s[0], axes[0, 1], "y=%i"%nearest_int(m_s[1]))
        plot_dot(source[:, :, nearest_int(m_s[0])], m_s[2], m_s[1], axes[0, 2], "x=%i"%nearest_int(m_s[0]))

        plot_dot(target[nearest_int(m_t[2]), :, :], m_t[1], m_t[0], axes[1, 0], "z=%i"%nearest_int(m_t[2]))
        axes[1, 0].set_ylabel("marker in target image.")
        plot_dot(target[:, nearest_int(m_t[1]), :], m_t[2], m_t[0], axes[1, 1], "y=%i"%nearest_int(m_t[1]))
        plot_dot(target[:, :, nearest_int(m_t[0])], m_t[2], m_t[1], axes[1, 2], "x=%i"%nearest_int(m_t[0]))

        plot_dot(target[nearest_int(m_w_t[2]), :, :], m_w_t[1], m_w_t[0], axes[2, 0], "z=%i"%nearest_int(m_w_t[2]))
        axes[2, 0].set_ylabel("warped marker in target image.")
        plot_dot(target[:, nearest_int(m_w_t[1]), :], m_w_t[2], m_w_t[0], axes[2, 1], "y=%i"%nearest_int(m_w_t[1]))
        plot_dot(target[:, :, nearest_int(m_w_t[0])], m_w_t[2], m_w_t[1], axes[2, 2], "x=%i"%nearest_int(m_w_t[0]))

        target = target.astype(np.int)
        source_itk = sitk.GetImageFromArray(source)
        target_itk = sitk.GetImageFromArray(target)
        overlay = sitk.LabelOverlay(source_itk, target_itk)
        axes[3,0].imshow(overlay[nearest_int(m_s[2]), :, :], cmap=plt.cm.gray)
        # axes[3,0].imshow(target[nearest_int(m_w_t[2]), :, :], cmap=plt.cm.gray)
        # plot_dot(warped[nearest_int(m_w[2]), :, :], m_w[1], m_w[0], axes[3, 0], "z=%i"%nearest_int(m_w[2]+origin[2]))
        # axes[3, 0].set_ylabel("warped marker in warped image.")
        # plot_dot(warped[:, nearest_int(m_w[1]), :], m_w[2], m_w[0], axes[3, 1], "y=%i"%nearest_int(m_w[1]+origin[1]))
        # plot_dot(warped[:, :, nearest_int(m_w[0])], m_w[2], m_w[1], axes[3, 2], "x=%i"%nearest_int(m_w[0]+origin[0]))

        plt.show()
        plt.savefig("./figure/lung_registration.png")
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)