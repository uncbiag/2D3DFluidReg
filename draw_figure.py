import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from evaluate_dir_lab import eval_with_file
import mermaid.module_parameters as pars

import os

def draw_reconstruction_figure(result_folder, idx_list):

    fig, axs = plt.subplots(1,3*len(idx_list))
    fig.subplots_adjust(wspace=0.2)
    cmap="gray"
    fontsize=4

    for i in range(len(idx_list)):
        ori = np.load(result_folder+"/"+idx_list[i]+'_origin.npy')
        rec = np.load(result_folder+"/"+idx_list[i]+'_rec.npy')

        mse = np.mean((ori-rec)**2)

        mid_id = int(ori.shape[1]/2)
        axs[i*3].imshow(ori[:,mid_id,:], cmap=cmap)
        axs[i*3].set_title("sDCT", fontsize=fontsize)
        axs[i*3].axis('off')
        # divider = make_axes_locatable(axs[i*3])
        # ax_cb = divider.new_horizontal(size="5%", pad=0.05)

        axs[i*3+1].imshow(rec[:,mid_id,:], cmap=cmap)
        axs[i*3+1].set_title("Reconstructed",fontsize=fontsize)
        axs[i*3+1].axis('off')

        axs[i*3+1].text(0.5,-0.15,"Patient %i MSE = %s"%(i,'{:.2e}'.format(mse)), size=fontsize, ha="center", transform=axs[i*3+1].transAxes)
        # divider = make_axes_locatable(axs[i*3+1])
        # ax_cb = divider.new_horizontal(size="5%", pad=0.05)

        res = axs[i*3+2].imshow(rec[:,mid_id,:]-ori[:,mid_id,:], vmin=np.min(ori[:,mid_id,:]), vmax=np.max(ori[:,mid_id,:]), cmap=cmap)
        axs[i*3+2].set_title("Residual error",fontsize=fontsize)
        axs[i*3+2].axis('off')

        # divider = make_axes_locatable(axs[i*3+2])
        # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        # cb = plt.colorbar(res, ax = axs[i*3+2], shrink=0.2, orientation='horizontal')
        # cb.ax.tick_params(labelsize=2)
    
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./figure/reconstruction.jpg", dpi=300, bbox_inches="tight",pad_inches=0.02)


# def draw_diff_figure(total,x,y,z,labels):
def prepare_data_for_diff(case_name, angle_list, proj_num_list, reg_model, setting, preprocessed_folder, disp_folder):
    lung_reg_params = pars.ParameterDict()
    total = []
    x = []
    y = []
    z = []
    angle = []
    proj_num = []
    for i in range(len(proj_num_list)):
        for j in range(len(angle_list)):
            lung_reg_params.load_JSON(setting)
            
            prefix = case_name+"_"+ str(angle_list[j]) + "_degree_"+str(proj_num_list[i])

            source_file = lung_reg_params["eval_marker_source_file"]
            target_file = lung_reg_params["eval_marker_target_file"]
            phi_file = disp_folder+"/" + prefix + "_"+reg_model+"_inverse_disp.npy"
            
            prop_file = preprocessed_folder + '/' + prefix +'_prop.npy'
            
            prop = np.load(prop_file, allow_pickle=True)
            dim = np.flip(np.array(prop.item().get("dim")))
            origin = np.flip(prop.item().get("crop")[0:3])
            spacing = np.flip(lung_reg_params["spacing"]).copy()

            res, res_seperate = eval_with_file(source_file, target_file, phi_file, dim, spacing, origin, False)
            total.append(res.item())
            x.append(res_seperate[0].item())
            y.append(res_seperate[1].item())
            z.append(res_seperate[2].item())
            angle.append(angle_list[j])
            proj_num.append(proj_num_list[i])
    return total, x, y, z, angle, proj_num

if __name__ == "__main__":
    # result_folder = './results/resonstruct'
    # idx_list = ["001","002","003"]
    # draw_reconstruction_figure(result_folder, idx_list)

    #############
    #############
    total, x, y, z, angle, proj_num = prepare_data_for_diff(
        "Case5Pack", [11,16,21,26,31,36], [4], 
        "affine","./lung_registration_setting_dct5.json", 
        "../eval_data/preprocessed_diff_angle", 
        "./results/exp_small_def_diff_angle")

    fig, ax = plt.subplots()
    ax.plot(angle, total, label="3D distance", marker='o')
    ax.plot(angle, z, label="X", marker='o')
    ax.plot(angle, x, label="Y", marker='o')
    ax.plot(angle, y, label="Z", marker='o')
    
    ax.set_xticks(angle)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xlabel("Scanning range of angles (degree)", fontsize=14)
    ax.set_ylabel("Distance (mm)", fontsize=14)
    ax.legend(loc=4, bbox_to_anchor=(1,0.0))
    ax.set_title("Affine registration \n with various scanning range", fontsize=16)
    plt.grid()
    plt.savefig("./figure/diff_angles_affine.png", dpi=300, bbox_inches="tight")
    plt.clf()
    
    #############
    #############
    total, x, y, z, angle, proj_num = prepare_data_for_diff(
        "Case5Pack", [11,16,21,26,31,36], [4], 
        "lddmm","./lung_registration_setting_dct5.json", 
        "../eval_data/preprocessed_diff_angle", 
        "./results/exp_small_def_diff_angle")

    fig, ax = plt.subplots()
    ax.plot(angle, total, label="3D distance", marker='o')
    ax.plot(angle, z, label="X", marker='o')
    ax.plot(angle, x, label="Y", marker='o')
    ax.plot(angle, y, label="Z", marker='o')
    
    
    ax.set_xticks(angle)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xlabel("Scanning range of angles (degree)", fontsize=14)
    ax.set_ylabel("Distance (mm)", fontsize=14)
    ax.legend(loc=4, bbox_to_anchor=(1,0))
    ax.set_title("LDDMM registration \n with various scanning range", fontsize=16)
    plt.grid()
    plt.savefig("./figure/diff_angles_lddmm.png", dpi=300,bbox_inches="tight")
    plt.clf()

    #############
    #############
    total, x, y, z, angle, proj_num = prepare_data_for_diff(
        "Case5Pack", [11], [4,6,8,10,12,14], 
        "affine","./lung_registration_setting_dct5.json", 
        "../eval_data/preprocessed_diff_angle", 
        "./results/exp_small_def_diff_proj_num")
    fig, ax = plt.subplots()
    ax.plot(proj_num, total, label="3D distance", marker='o')
    ax.plot(proj_num, z, label="X", marker='o')
    ax.plot(proj_num, x, label="Y", marker='o')
    ax.plot(proj_num, y, label="Z", marker='o')
    
    ax.set_xticks(proj_num)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xlabel("Number of projections", fontsize=14)
    ax.set_ylabel("Distance (mm)", fontsize=14)
    ax.legend(loc=4, bbox_to_anchor=(1,0.0))
    ax.set_title("Affine registration \n with various projection number", fontsize=16)
    plt.grid()
    plt.savefig("./figure/diff_proj_affine.png", dpi=300,bbox_inches="tight")
    plt.clf()

    #############
    #############
    total, x, y, z, angle, proj_num = prepare_data_for_diff(
        "Case5Pack", [11], [4,6,8,10,12,14], 
        "lddmm","./lung_registration_setting_dct5.json", 
        "../eval_data/preprocessed_diff_angle", 
        "./results/exp_small_def_diff_proj_num")
    fig, ax = plt.subplots()
    ax.plot(proj_num, total, label="3D distance", marker='o')
    ax.plot(proj_num, z, label="X", marker='o')
    ax.plot(proj_num, x, label="Y", marker='o')
    ax.plot(proj_num, y, label="Z", marker='o')
    
    ax.set_xticks(proj_num)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xlabel("Number of projections", fontsize=14)
    ax.set_title("LDDMM registration \n with various projection number", fontsize=16)
    ax.set_ylabel("Distance (mm)", fontsize=14)
    ax.legend(loc=4, bbox_to_anchor=(1,0.0))
    plt.grid()
    plt.savefig("./figure/diff_proj_lddmm.png", dpi=300,bbox_inches="tight")
    plt.clf()