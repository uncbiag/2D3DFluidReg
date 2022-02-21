import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from reconstruction_models.layers import projection
from reconstruction_models.reconBase import reconBase
from utils.plot_helper import show_image_with_colorbar
from utils.image_utils import image_gradient_3d

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class recon_model(reconBase):
    def run(self, I_proj, I_target, I_source=None, I_seg=None, I_warped=None, I_proj_orth=None, id=""):
        if self.log_step > 0:
            log_path = self.log_path + "/" + id
            if not os.path.exists(log_path):
                os.mkdir(log_path)
        
        resolution = [I_proj.shape[2], I_proj.shape[3]]
        sample_rate = [int(1), int(1), int(1)]
        I_target_npy = I_target.cpu().numpy()[0,0]
        model = projection(I_target.shape, resolution, sample_rate, self.spacing, None, I_seg, I_proj.device)
        model.to(I_proj.device)
        I_seg = None
        opt = torch.optim.Adam(model.parameters(), lr=0.6)
        # opt = torch.optim.Adam(model.parameters(), lr=0.0008)
        # opt = torch.optim.Adam(model.parameters(), lr=0.0005)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3, cooldown=5, verbose=True)

        loss_log = []
        for i in range(self.epochs):
            opt.zero_grad()
            output = model(self.poses)
            # miccai reconstruction
            loss = 200*self._nccSim(output, I_proj, 'mean') + 0.1*self._sim(output, I_proj, 'mean')
            # loss = 200*sim(output, I_proj)

            # TODO: remove later. Debug code to show gradient
            # if i>100:
            #     loss.backward(retain_graph=True)
            #     for p in model.parameters():
            #         slices = p.grad.detach().cpu().numpy()[0,0]
            #         plt.clf()
            #         fig, axes = plt.subplots(1,10)
            #         for j in range(10):
            #             axes[j].imshow(slices[:,70+3*j,:])
            #         plt.savefig(os.path.join(log_path,"gradient_of_proj_loss_%i.png"%i), dpi=300)
            #     opt.zero_grad()

            total_loss = loss
            for p in model.parameters():
                # setting for CT
                # reg_loss = 10000*torch.sum(F.relu(-1*p))# + torch.mean(image_gradient_3d(p, device))
                # Miccai reconstruction
                reg_loss = torch.mean((p)**2) + 5*torch.sum(F.relu(-1*p))
                total_loss = total_loss + reg_loss
                # setting for sDCT
                # total_loss = total_loss + 1000*torch.sum(F.relu(-1*p)) + torch.mean(image_gradient_3d(p, device))
                
                if I_warped is not None:
                    if I_seg is not None:
                        sim_loss_3d = 1000*self._nccSim(I_warped*I_seg, p*I_seg)
                    else:
                        sim_loss_3d = self._sim(I_warped, p, 'mean')
                        
                        # TODO: remove later. debug code to show gradient
                        # if i>100:
                        #     sim_loss_3d.backward(retain_graph=True)
                        #     for p in model.parameters():
                        #         slices = p.grad.detach().cpu().numpy()[0,0]
                        #         plt.clf()
                        #         fig, axes = plt.subplots(1,10)
                        #         for j in range(10):
                        #             axes[j].imshow(slices[:,70+3*j,:])
                        #         plt.savefig(os.path.join(log_path,"gradient_of_3d_loss_%i.png"%i), dpi=300)
                        #     opt.zero_grad()
                    total_loss = total_loss + sim_loss_3d  #800*((I_warped - p) ** 2).mean()
                else:
                    if I_seg is not None:
                        total_loss = total_loss + 100*torch.mean(p*(1-I_seg))
                    else:
                        sim_loss_3d = torch.tensor(0)

                if I_proj_orth is not None:
                    I_rec_proj_orth = torch.sum(p,4)
                    orth_loss = self._sim(I_proj_orth, I_rec_proj_orth, reduction='mean')
                    total_loss = total_loss + orth_loss
                else: 
                    orth_loss = torch.tensor(0)

            
            # record the loss
            loss_log.append([total_loss.item(), loss.item(), reg_loss.item(), sim_loss_3d.item(), orth_loss.item()])

            #early stop
            if total_loss.data < 1e-4:
                break
            total_loss.backward()
            opt.step()
            scheduler.step(total_loss)

            if self.log_step > 0:
                if i%self.log_step == self.log_step-1:
                    for p in model.parameters():
                        # Plot I_r and I_target
                        slice_count = 5
                        reconstructed = p.detach().cpu().numpy()[0,0]
                        grad = p.grad.cpu().numpy()[0,0]
                        step = int(reconstructed.shape[1]/slice_count)

                        fig, axs = plt.subplots(3,slice_count)
                        for j in range(0,slice_count):
                            show_image_with_colorbar(reconstructed[:,j*step,:], axs[0,j], 
                                            shrink=1.0, labelsize=4)
                            show_image_with_colorbar(I_target_npy[:,j*step,:], axs[1,j], 
                                            shrink=1.0, labelsize=4)
                            show_image_with_colorbar(grad[:,j*step,:], axs[2,j],
                                                    shrink=1.0, labelsize=4)
                            
                        for ax in axs:
                            for x in ax:
                                x.set_xticks([])
                                x.set_yticks([])

                        axs[0,0].set_ylabel("I_R", rotation=0, ha="right")
                        axs[1,0].set_ylabel("I_truth", rotation=0, ha="right")
                        
                        fig.suptitle("Iteration = %i"%i)
                        fig.tight_layout()
                        plt.savefig(os.path.join(log_path, "reconstructed_full_%i.png"%i), dpi=300)
                        
                        # Plot projections of I_r and I_target
                        if I_proj_orth is not None:
                            fig, axs = plt.subplots(2,5)
                            for j in range(0,4):
                                show_image_with_colorbar(output[0,j].detach().cpu().numpy(),
                                                            axs[0,j], shrink=0.5,
                                                            pad=0.02, labelsize=5)
                                show_image_with_colorbar(I_proj[0,:,:,:][j].cpu().numpy(),
                                                            axs[1,j], shrink=0.5,
                                                            pad=0.02, labelsize=5)
                            show_image_with_colorbar(I_rec_proj_orth[0,0].detach().cpu().numpy(),
                                                        axs[0,4], shrink=0.5,
                                                        pad=0.02, labelsize=5)
                            show_image_with_colorbar(I_proj_orth[0,0].cpu().numpy(),
                                                        axs[1,4], shrink=0.5,
                                                        pad=0.02, labelsize=5)
                            plt.savefig(os.path.join(log_path,"projection_%i.png"%i), dpi=300)
                        else:
                            fig, axs = plt.subplots(2,4)
                            for j in range(0,4):
                                show_image_with_colorbar(output[0,j].detach().cpu().numpy(),
                                                            axs[0,j], shrink=1.0,
                                                            pad=0.01, labelsize=4)
                                show_image_with_colorbar(I_proj[0,:,:,:][j].cpu().numpy(),
                                                            axs[1,j], shrink=1.0,
                                                            pad=0.01, labelsize=4)
                            
                            for ax in axs:
                                for x in ax:
                                    x.set_xticks([])
                                    x.set_yticks([])

                            axs[0,0].set_ylabel("Projections \n of I_R", rotation=0, ha="right")
                            axs[1,0].set_ylabel("Projections \n of I_truth", rotation=0, ha="right")
                            fig.suptitle("Iteration = %i"%i)
                            fig.tight_layout()
                            plt.savefig(os.path.join(log_path,"projection_%i.png"%i), dpi=300)
                
                    loss_at_it = loss_log[i]
                    proj_ncc_loss = self._nccSim(output, I_proj)
                    print("Epoch %i: total loss: %f, rec loss: %f, reg loss: %f, orth loss: %f, 3D sim loss: %f | proj ncc: %f, 3D sim ncc: %f"%(i, loss_at_it[0], loss_at_it[1], loss_at_it[2], loss_at_it[4], loss_at_it[3], proj_ncc_loss.data, loss_at_it[3]/100))# loss.data, reg_loss.data, orth_loss.data, sim_loss_3d.data))

        # At the end of optimization, plot output
        for p in model.parameters():
            loss_at_it = loss_log[i]
            proj_ncc_loss = self._nccSim(output, I_proj)
            print("Epoch %i: total loss: %f, rec loss: %f, reg loss: %f, orth loss: %f, 3D sim loss: %f | proj ncc: %f, 3D sim ncc: %f"%(i, loss_at_it[0], loss_at_it[1], loss_at_it[2], loss_at_it[4], loss_at_it[3], proj_ncc_loss.data, loss_at_it[3]/100))

            # Save reconstructed volume
            reconstructed = p.detach().cpu().numpy()[0,0]
            np.save(self.result_path+"/"+id+"_recon.py", reconstructed)

            # Plot results
            if I_warped is not None:
                I_warped_npy = I_warped.cpu().numpy()[0,0]
            else:
                I_warped_npy = None

            self._plot_rec_result(reconstructed, I_target_npy, 
                            os.path.join(self.log_path, id + "_reconstructed_diff.png"), 
                            I_warped=I_warped_npy)
            # print_sim_of_all(I_target, I_source, p, output, pose_scale, sample_rate, spacing, 
            #              poses, I_warped=None, save_path="")

        # plot the loss
        # plt.clf()
        # loss_log_npy = np.array(loss_log)
        # fig, axes = plt.subplots(3,2)
        # label = ['total', 'rec', 'reg', '3D Sim', 'orth']
        # for i in range(5):
        #     axes[int(i/2),i%2].plot(loss_log_npy[:,i])
        #     axes[int(i/2),i%2].set_xlabel(label[i])
        # fig.tight_layout()
        # plt.savefig(os.path.join(self.log_path, "loss_record.png"), dpi=300)
