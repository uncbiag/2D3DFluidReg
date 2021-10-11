import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot_helper import show_image_with_colorbar
from common.preprocessing import calculate_projection


class reconBase:
    def __init__(self, poses, spacing, epochs=400, log_step=100, log_path = "./log", result_path="./result"):
        self.poses = poses
        self.spacing = spacing
        self.epochs = epochs
        self.log_step = log_step
        self.log_path = log_path
        self.result_path = result_path

    def _sim(self, I0, I1, reduction='mean'):
        # return l1_loss(I0, I1, reduction='mean')
        # torch.mean(torch.norm(output-I_proj[idx,:,:,:], dim=(2,3)))
        if reduction == 'mean':
            return ((I0-I1)**2).mean()
        elif reduction == 'sum':
            channel = I0.shape[0]
            return ((I0-I1)**2).sum()


    def _nccSim(self, I0, I1, reduction='mean'):
        dim = len(I0.shape[2:])
        channel = I0.shape[1]
        input_shape = [I0.shape[0], I0.shape[1], -1]+[1]*dim
        I0 = I0.view(*input_shape)
        I1 = I1.view(*input_shape)  
        I0mean = I0.mean(dim=2, keepdim=True)
        I1mean = I1.mean(dim=2, keepdim=True)
        I0_m_mean = I0-I0mean
        I1_m_mean = I1-I1mean
        nccSqr = ((((I0_m_mean)*(I1_m_mean)).mean(dim=2)**2)+1e-12)/\
                    ((((I0_m_mean)**2).mean(dim=2)*((I1_m_mean)**2).mean(dim=2))+1e-12)
        nccSqr =nccSqr.sum()
        if reduction == 'mean':
            return 1.-nccSqr/channel
        elif reduction == 'sum': 
            return channel*1.-nccSqr


    def print_sim_of_all(self, I_target, I_source, I_rec, I_rec_proj, pose_scale, sample_rate, spacing, 
                        poses, I_warped=None, save_path=""):
        p_target = torch.from_numpy(calculate_projection(I_target[0,0].cpu().numpy(), 
                                    pose_scale, 1.4,
                                    sample_rate, spacing, I_target.device)).to(I_target.device).unsqueeze(1)
        p_source = torch.from_numpy(calculate_projection(I_source[0,0].cpu().numpy(), pose_scale, 1.4,
                                        sample_rate, spacing, I_target.device)).to(I_target.device).unsqueeze(1)
        p_r = torch.from_numpy(calculate_projection(I_rec[0,0].detach().cpu().numpy(), pose_scale, 1.4,
                                        sample_rate, spacing, I_target.device)).to(I_target.device).unsqueeze(1)

        # p_target = torch.sum(I_target, axis=2)
        # p_source = torch.sum(I_source, axis=2)
        # p_r = torch.sum(I_rec, axis=2)
        sim_all = []

        r_target_ncc = nccSim(I_rec, I_target)
        r_target_ssd = torch.mean((I_rec-I_target)**2)
        source_target_ncc = nccSim(I_source, I_target)
        source_target_ssd = torch.mean((I_source-I_target)**2)
        source_target_proj_ncc = nccSim(p_source, p_target)
        source_target_proj_ssd = torch.mean((p_source-p_target)**2)
        r_target_proj_ncc = nccSim(I_rec_proj, p_target)
        r_target_proj_ssd = torch.mean((I_rec_proj-p_target)**2)
        print("3D   - I_r      vs I_target (ncc, ssd):%10.3e, %10.3e"%(r_target_ncc.data, r_target_ssd.data))
        print("3D   - I_source vs I_target (ncc, ssd):%10.3e, %10.3e"%(source_target_ncc.data, source_target_ssd.data))
        print("Proj - I_r      vs I_target (ncc, ssd):%10.3e, %10.3e"%(r_target_proj_ncc.data, r_target_proj_ssd.data))
        print("Proj - I_source vs I_target (ncc, ssd):%10.3e, %10.3e"%(source_target_proj_ncc.data, source_target_proj_ssd.data))
        sim_all = [["I_r vs I_target", r_target_ncc.data, r_target_ssd.data, r_target_proj_ncc.data, r_target_proj_ssd.data],
                ["I_source vs I_target", source_target_ncc.data, source_target_ssd.data, source_target_proj_ncc.data, source_target_proj_ssd.data]]

        if I_warped is not None:
            p_warped = torch.from_numpy(calculate_projection(I_warped[0,0].cpu().numpy(), pose_scale, 1.4,
                                            sample_rate, spacing, I_target.device)).to(I_target.device).unsqueeze(1)

            # p_warped = torch.sum(I_warped, axis=2)

            sim1 = nccSim(I_rec, I_warped)
            sim3 = nccSim(I_warped, I_target)
            sim4 = torch.mean((I_rec-I_warped)**2)
            sim6 = torch.mean((I_warped-I_target)**2)
            warped_target_proj_ncc = nccSim(p_warped, p_target)
            warped_target_proj_ssd = torch.mean((p_warped-p_target)**2)
            warted_rec_proj_ncc = nccSim(p_warped, p_r)
            warped_rec_proj_ssd = torch.mean((p_warped-p_r)**2)
            print("3D   - I_r      vs I_warped (ncc, ssd):%10.3e, %10.3e"%(sim1.data, sim4.data))
            print("3D   - I_warped vs I_target (ncc, ssd):%10.3e, %10.3e"%(sim3.data, sim6.data))
            print("Proj - I_warped vs I_target (ncc, ssd):%10.3e, %10.3e"%(warped_target_proj_ncc.data, warped_target_proj_ssd.data))
            sim_all.append(["I_warped vs I_target", sim3.data, sim6.data, warped_target_proj_ncc.data, warped_target_proj_ssd.data])
            sim_all.append(["I_r vs I_warped", sim1.data, sim4.data, warted_rec_proj_ncc.data, warped_rec_proj_ssd.data])

        if save_path != "":
            df = pd.DataFrame(sim_all)
            df.to_csv(save_path)


    def _plot_rec_result(self, I_rec, I_target, save_path, I_warped=None):
        """
        :param I_rec: reconstructed image. Numpy array.
        :param I_target: target image. Numpy array.
        :param save_path: path of the saved figure.
        :param I_warped: warped image coming out of a registration algorithm. Numpy array.
        """
        step = int(I_rec.shape[1]/5)

        # Plot I_r, I_reconstruct
        diff = I_rec-I_target
        vmin = np.min(diff)
        vmax = np.max(diff)
        img_min = min(np.min(I_target), np.min(I_rec))
        img_max = max(np.max(I_target), np.max(I_rec))
        fig, axs = plt.subplots(5,5)
        for j in range(0,5):
            if I_warped is not None:
                show_image_with_colorbar((I_rec-I_warped)[:,j*step,:], axs[0,j])
                show_image_with_colorbar(I_warped[:,j*step,:], axs[4,j])
            else:
                fig.delaxes(axs[0,j])
                fig.delaxes(axs[4,j])

            show_image_with_colorbar((diff)[:,j*step,:], axs[1,j], shrink=0.65, vmin=vmin, vmax=vmax)
            axs[2,j].imshow(I_target[:,j*step,:], vmin=img_min, vmax=img_max)
            axs[3,j].imshow(I_rec[:,j*step,:], vmin=img_min, vmax=img_max)
            # show_image_with_colorbar(I_target[:,j*step,:], axs[2,j], vmin=img_min, vmax=img_max)
            # show_image_with_colorbar(I_rec[:,j*step,:], axs[3,j], vmin=img_min, vmax=img_max)

        # Hide ticks
        for ax in axs:
            for x in ax:
                x.set_xticks([])
                x.set_yticks([])
        
        # Add label for each row
        axs[0,0].set_ylabel("Differences between \n I_R and I_warp", rotation=0, ha="right", ma="center", fontsize=4)
        axs[1,0].set_ylabel("Differences between \n I_R and I_truth", rotation=0, ha="right",ma="center",fontsize=4)
        axs[2,0].set_ylabel("I_truth", rotation=0, ha="right",ma="center",fontsize=4)
        axs[3,0].set_ylabel("I_R", rotation=0, ha="right", ma="center",fontsize=4)
        axs[4,0].set_ylabel("I_warp", rotation=0, ha="right", ma="center",fontsize=4)

        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path, dpi=300)


    def run(self):
        raise NotImpementedError