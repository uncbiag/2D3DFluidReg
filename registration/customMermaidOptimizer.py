import os
import time
import numpy as np
import torch
from termcolor import colored, cprint
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import random

from mermaid import utils
from mermaid import model_factory as MF
from mermaid import custom_optimizers as CO
from mermaid.data_wrapper import USE_CUDA, AdaptVal
from mermaid import image_sampling as IS
from mermaid import visualize_registration_results as vizReg
from mermaid.data_utils import make_dir
from mermaid import fileio as FIO
from mermaid import model_evaluation

from mermaid.multiscale_optimizer import ImageRegistrationOptimizer
from mermaid.multiscale_optimizer import SimpleRegistration

from mermaid.similarity_measure_factory import NCCSimilarity, SSDSimilarity


class CustomSimpleSingleScaleRegistration(SimpleRegistration):
    """
    Simple single scale registration
    """
    def __init__(self,ISource,ITarget,ITargetProj,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(CustomSimpleSingleScaleRegistration, self).__init__(ISource,ITarget,spacing,sz,params,compute_inverse_map=compute_inverse_map,default_learning_rate=default_learning_rate)
        self.ITarget = nn.Parameter(self.ITarget, requires_grad=True)
        self.optimizer = CustomSingleScaleRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,ITargetProj,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource, self.ITarget)


def image_gradient(x):
    device = x.device
    
    fil = torch.tensor([[1,2,1],[2,4,2],[1,2,1]])
    x_filter = torch.zeros((3,3,3), device=device).view(1,1,3,3,3)
    x_filter[0,0,0,:,:]=fil
    x_filter[0,0,2,:,:]=-fil
    g_x = F.conv3d(x, x_filter, padding=1)

    y_filter = torch.zeros((3,3,3), device=device).view(1,1,3,3,3)
    y_filter[0,0,:,0,:]=fil
    y_filter[0,0,:,2,:]=-fil
    g_y = F.conv3d(x, y_filter, padding=1)

    z_filter = torch.zeros((3,3,3), device=device).view(1,1,3,3,3)
    z_filter[0,0,:,:,0]=fil
    z_filter[0,0,:,:,2]=-fil
    g_z = F.conv3d(x, z_filter, padding=1)

    return torch.norm(torch.cat((g_x, g_y, g_z), 0), dim=0)


class projection(nn.Module):
    def __init__(self, resolution, sample_rate, spacing, emi_poses_scale, device):
        super(projection, self).__init__()

        self.resolution = resolution
        self.sample_rate = sample_rate
        self.spacing = torch.from_numpy(spacing).to(device)
        self.batch_size = 4
        self.batch_start = 0
        self.emi_poses_scale = emi_poses_scale

    def forward(self, IReconstruct):
        poses = self.emi_poses_scale*IReconstruct.shape[3]
        device = IReconstruct.device
        # projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
        projections = torch.zeros((1, len(poses), int(self.resolution[0]*self.sample_rate[0]), int(self.resolution[1]*self.sample_rate[2]))).to(device)
        # idx = random.sample(list(range(poses.shape[0])), self.batch_size)
        # idx = range(0, poses.shape[0])
        # idx = range(self.batch_start, min(self.batch_start+self.batch_size, poses.shape[0]))
        # if self.batch_start+self.batch_size > poses.shape[0]:
        #     self.batch_start = 0
        # else:
        #     self.batch_start += self.batch_size
        for i in range(len(poses)):
            grid, dx = self.project_grid(IReconstruct, poses[i], (self.resolution[0], self.resolution[1]), self.sample_rate, IReconstruct.shape[2:])
            grid = torch.flip(grid, [3])
            dx = dx.unsqueeze(0).unsqueeze(0)
            projections[0,i] = torch.mul(torch.sum(F.grid_sample(IReconstruct, grid.unsqueeze(0), align_corners=True), dim=4), dx)[0, 0]
            del grid
            torch.cuda.empty_cache()
            
        return projections

    def project_grid(self, img, emi_pos, resolution, sample_rate, obj_shape):
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
        dx = torch.norm(dx*self.spacing.unsqueeze(0).unsqueeze(0), dim=2)
        # dx = torch.abs(torch.mul(torch.ones((I.shape[0],I.shape[1]), device=device),1./I[:,:,1]))

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

class CustomSingleScaleRegistrationOptimizer(ImageRegistrationOptimizer):
    """
    Optimizer operating on a single scale. Typically this will be the full image resolution.

    .. todo::
        Check what the best way to adapt the tolerances for the pre-defined optimizers;
        tying it to rel_ftol is not really correct.
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, ITargetProj, compute_inverse_map=False, default_learning_rate=None):
        super(CustomSingleScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

        if self.mapLowResFactor is not None:
            # computes model at a lower resolution than the image similarity
            if self.compute_similarity_measure_at_low_res:
                self.mf = MF.ModelFactory(self.lowResSize, self.lowResSpacing, self.lowResSize, self.lowResSpacing )
            else:
                self.mf = MF.ModelFactory(self.sz, self.spacing, self.lowResSize, self.lowResSpacing )
        else:
            # computes model and similarity at the same resolution
            self.mf = MF.ModelFactory(self.sz, self.spacing, self.sz, self.spacing)
        """model factory which will be used to create the model and its loss function"""

        self.model = None
        """the model itself"""
        self.criterion = None
        """the loss function"""

        self.initialMap = None
        """initial map, will be needed for map-based solutions; by default this will be the identity map, but can be set to something different externally"""
        self.initialInverseMap = None
        """initial inverse map; will be the same as the initial map, unless it was set externally"""
        self.map0_inverse_external = None
        """initial inverse map, set externally, will be needed for map-based solutions; by default this will be the identity map, but can be set to something different externally"""
        self.map0_external = None
        """intial map, set externally"""
        self.lowResInitialMap = None
        """low res initial map, by default the identity map, will be needed for map-based solutions which are computed at lower resolution"""
        self.lowResInitialInverseMap = None
        """low res initial inverse map, by default the identity map, will be needed for map-based solutions which are computed at lower resolution"""
        self.weight_map =None
        """init_weight map, which only used by metric learning models"""
        self.optimizer_instance = None
        """the optimizer instance to perform the actual optimization"""

        c_params = self.params[('optimizer', {}, 'optimizer settings')]
        self.weight_clipping_type = c_params[('weight_clipping_type','none','Type of weight clipping that should be used [l1|l2|l1_individual|l2_individual|l1_shared|l2_shared|None]')]
        self.weight_clipping_type = self.weight_clipping_type.lower()
        """Type of weight clipping; applied to weights and bias indepdenendtly; norm restricted to weight_clipping_value"""
        if self.weight_clipping_type=='none':
            self.weight_clipping_type = None
        if self.weight_clipping_type!='pre_lsm_weights':
            self.weight_clipping_value = c_params[('weight_clipping_value', 1.0, 'Value to which the norm is being clipped')]
            """Desired norm after clipping"""

        extent = self.spacing * self.sz[2:]
        max_extent = max(extent)

        clip_params = c_params[('gradient_clipping',{},'clipping settings for the gradient for optimization')]
        self.clip_display = clip_params[('clip_display', True, 'If set to True displays if clipping occurred')]
        self.clip_individual_gradient = clip_params[('clip_individual_gradient',False,'If set to True, the gradient for the individual parameters will be clipped')]
        self.clip_individual_gradient_value = clip_params[('clip_individual_gradient_value',max_extent,'Value to which the gradient for the individual parameters is clipped')]
        self.clip_shared_gradient = clip_params[('clip_shared_gradient', True, 'If set to True, the gradient for the shared parameters will be clipped')] # todo recover the clip gradient,or it may cause unstable
        self.clip_shared_gradient_value = clip_params[('clip_shared_gradient_value', 1.0, 'Value to which the gradient for the shared parameters is clipped')]

        self.scheduler = None # for the step size scheduler
        self.patience = None # for the step size scheduler
        self._use_external_scheduler = False

        self.rec_energy = None
        self.rec_similarityEnergy = None
        self.rec_regEnergy = None
        self.rec_opt_par_loss_energy = None
        self.rec_phiWarped = None
        self.rec_phiInverseWarped = None
        self.rec_IWarped = None
        self.last_energy = None
        self.rel_f = None
        self.rec_custom_optimizer_output_string = ''
        """the evaluation information"""
        self.rec_custom_optimizer_output_values = None

        self.delayed_model_parameters = None
        self.delayed_model_parameters_still_to_be_set = False
        self.delayed_model_state_dict = None
        self.delayed_model_state_dict_still_to_be_set = False

        # to be able to transfer state and parameters
        self._sgd_par_list = None # holds the list of parameters
        self._sgd_par_names = None # holds the list of names associated with these parameters
        self._sgd_name_to_model_par = None # allows mapping from name to model parameter
        self._sgd_split_shared = None # keeps track if the shared states were split or not
        self._sgd_split_individual = None # keeps track if the individual states were split or not
        self.over_scale_iter_count = None #accumulated iter count over different scales
        self.n_scale = None #the index of  current scale, torename and document  todo

        # For reconstruction
        self.ITargetProj = ITargetProj
        resolution = [ITargetProj.shape[2], ITargetProj.shape[3]]
        proj_spacing = np.array(self.params['model']['registration_model']["similarity_measure"]["projection"]["spacing"])
        sample_rate = self.params['model']['registration_model']["sample_rate"]
        emi_poses_scale = np.array(self.params['model']['registration_model']['emitter_pos_scale_list'])
        self.recon_sigma = self.params['model']['registration_model']["similarity_measure"]["projection"]["sigma"]
        self.recon_volumn = resolution[0]*resolution[1]*ITargetProj.shape[1]
        self.rec_recon_energy = None
        self.recon_lr = self.params['model']['registration_model']["similarity_measure"]["projection"]["lr"]
        
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.projection = projection(resolution, sample_rate, proj_spacing, emi_poses_scale, device)

        sim_spacing = 1./(np.array(self.ITargetProj.shape[2:]).astype(np.int16) - 1)
        self.proj_sim_measure = SSDSimilarity(sim_spacing, self.params)
        self.proj_sim_measure.sigma = self.recon_sigma

    def write_parameters_to_settings(self):
        if self.model is not None:
            self.model.write_parameters_to_settings()

    def get_sgd_split_shared(self):
        return self._sgd_split_shared

    def get_sgd_split_indvidual(self):
        return self._sgd_split_individual

    def get_checkpoint_dict(self):
        if self.model is not None and self.optimizer_instance is not None:
            d = super(CustomSingleScaleRegistrationOptimizer, self).get_checkpoint_dict()
            d['model'] = dict()
            d['model']['parameters'] = self.model.get_registration_parameters_and_buffers()
            d['model']['size'] = self.model.sz
            d['model']['spacing'] = self.model.spacing
            d['optimizer_state'] = self.optimizer_instance.state_dict()
            return d
        else:
            raise ValueError('Unable to create checkpoint, because either the model or the optimizer have not been initialized')

    def load_checkpoint_dict(self,d,load_optimizer_state=False):
        if self.model is not None and self.optimizer_instance is not None:
            self.model.set_registration_parameters(d['model']['parameters'],d['model']['size'],d['model']['spacing'])
            if load_optimizer_state:
                try:
                    self.optimizer_instance.load_state_dict(d['optimizer_state'])
                    print('INFO: Was able to load the previous optimzer state from checkpoint data')
                except:
                    print('INFO: Could not load the previous optimizer state')
            else:
                print('WARNING: Turned off the loading of the optimizer state')
        else:
            raise ValueError('Cannot load checkpoint dictionary, because either the model or the optimizer have not been initialized')

    def get_opt_par_energy(self):
        """
        Energy for optimizer parameters

        :return:
        """
        return self.rec_opt_par_loss_energy.cpu().item()

    def get_custom_output_values(self):
        """
        Custom output values

        :return:
        """
        return self.rec_custom_optimizer_output_values

    def get_energy(self):
        """
        Returns the current energy
        :return: Returns a tuple (energy, similarity energy, regularization energy)
        """
        return self.rec_energy.cpu().item(), self.rec_similarityEnergy.cpu().item(), self.rec_regEnergy.cpu().item()

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.useMap:
            cmap = self.get_map()
            # and now warp it
            return utils.compute_warped_image_multiNC(self.ISource, cmap, self.spacing, self.spline_order,zero_boundary=True)
        else:
            return self.rec_IWarped

    def get_warped_label(self):
        """
        Returns the warped label
        :return: the warped label
        """
        if self.useMap:
            cmap = self.get_map()
            return utils.get_warped_label_map(self.LSource, cmap, self.spacing)
        else:
            return None

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        return self.rec_phiWarped

    def get_inverse_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        return self.rec_phiInverseWarped

    def set_n_scale(self, n_scale):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        self.n_scale = n_scale

    def set_over_scale_iter_count(self, iter_count):
        self.over_scale_iter_count = iter_count


    def _create_initial_maps(self):
        if self.useMap:
            # create the identity map [-1,1]^d, since we will use a map-based implementation
            if self.map0_external is not None:
                self.initialMap = self.map0_external
            else:
                id = utils.identity_map_multiN(self.sz, self.spacing)
                self.initialMap = AdaptVal(torch.from_numpy(id))

            if self.map0_inverse_external is not None:
                self.initialInverseMap = self.map0_inverse_external
            else:
                id =utils.identity_map_multiN(self.sz, self.spacing)
                self.initialInverseMap =  AdaptVal(torch.from_numpy(id))

            if self.mapLowResFactor is not None:
                # create a lower resolution map for the computations
                if self.map0_external is None:
                    lowres_id = utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                    self.lowResInitialMap = AdaptVal(torch.from_numpy(lowres_id))
                else:
                    sampler = IS.ResampleImage()
                    lowres_id, _ = sampler.downsample_image_to_size(self.initialMap , self.spacing,self.lowResSize[2::] , 1,zero_boundary=False)
                    self.lowResInitialMap = AdaptVal(lowres_id)

                if self.map0_inverse_external is None:
                    lowres_id = utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                    self.lowResInitialInverseMap = AdaptVal(torch.from_numpy(lowres_id))
                else:
                    sampler = IS.ResampleImage()
                    lowres_inverse_id, _ = sampler.downsample_image_to_size(self.initialInverseMap, self.spacing, self.lowResSize[2::],
                                                                    1, zero_boundary=False)
                    self.lowResInitialInverseMap = AdaptVal(lowres_inverse_id)

    def set_model(self, modelName):
        """
        Sets the model that should be solved

        :param modelName: name of the model that should be solved (string)
        """

        self.params['model']['registration_model']['type'] = ( modelName, "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with '_map' or '_image' suffix" )

        self.model, self.criterion = self.mf.create_registration_model(modelName, self.params['model'],compute_inverse_map=self.compute_inverse_map)
        print(self.model)

        self._create_initial_maps()

    def set_initial_map(self,map0,map0_inverse=None):
        """
        Sets the initial map (overwrites the default identity map)
        :param map0: intial map
        :param map0_inverse: initial inverse map
        :return: n/a
        """

        self.map0_external = map0
        self.map0_inverse_external = map0_inverse

        if self.initialMap is not None:
            # was already set, so let's modify it
            self._create_initial_maps()

    def set_initial_weight_map(self,weight_map,freeze_weight=False):
        """
        Sets the initial map (overwrites the default identity map)
        :param map0: intial map
        :param map0_inverse: initial inverse map
        :return: n/a
        """
        if self.mapLowResFactor is not None:
            sampler = IS.ResampleImage()
            weight_map, _ = sampler.downsample_image_to_size(weight_map, self.spacing, self.lowResSize[2::], 1,
                                                            zero_boundary=False)
        self.model.local_weights.data = weight_map
        if freeze_weight:
            self.model.freeze_adaptive_regularizer_param()

    def get_initial_map(self):
        """
        Returns the initial map

        :return: initial map
        """

        if self.initialMap is not None:
            return self.initialMap
        elif self.map0_external is not None:
            return self.map0_external
        else:
            return None

    def get_initial_inverse_map(self):
        """
        Returns the initial inverse map

        :return: initial inverse map
        """

        if self.initialInverseMap is not None:
            return self.initialInverseMap
        elif self.map0_inverse_external is not None:
            return self.map0_inverse_external
        else:
            return None

    def add_similarity_measure(self, sim_name, sim_measure):
        """
        Adds a custom similarity measure.

        :param sim_name: name of the similarity measure (string)
        :param sim_measure: similarity measure itself (class object that can be instantiated)
        """
        self.criterion.add_similarity_measure(sim_name, sim_measure)
        self.params['model']['registration_model']['similarity_measure']['type'] = (sim_name, 'was customized; needs to be expplicitly instantiated, cannot be loaded')

    def add_model(self, model_name, model_network_class, model_loss_class, use_map, model_description='custom model'):
        """
        Adds a custom model and its loss function

        :param model_name: name of the model to be added (string)
        :param model_network_class: registration model itself (class object that can be instantiated)
        :param model_loss_class: registration loss (class object that can be instantiated)
        :param use_map: True/False: specifies if model uses a map or not
        :param model_description: optional model description
        """
        self.mf.add_model(model_name, model_network_class, model_loss_class, use_map, model_description)
        self.params['model']['registration_model']['type'] = (model_name, 'was customized; needs to be explicitly instantiated, cannot be loaded')

    def set_model_state_dict(self,sd):
        """
        Sets the state dictionary of the model

        :param sd: state dictionary
        :return: n/a
        """

        if self.optimizer_has_been_initialized:
            self.model.load_state_dict(sd)
            self.delayed_model_state_dict_still_to_be_set = False
        else:
            self.delayed_model_state_dict_still_to_be_set = True
            self.delayed_model_state_dict = sd

    def get_model_state_dict(self):
        """
        Returns the state dictionary of the model

        :return: state dictionary
        """
        return self.model.state_dict()

    def set_model_parameters(self, p):
        """
        Set the parameters of the registration model

        :param p: parameters
        """

        if self.optimizer_has_been_initialized:
            if (self.useMap) and (self.mapLowResFactor is not None):
                self.model.set_registration_parameters(p, self.lowResSize, self.lowResSpacing)
            else:
                self.model.set_registration_parameters(p, self.sz, self.spacing)
            self.delayed_model_parameters_still_to_be_set = False
        else:
            self.delayed_model_parameters_still_to_be_set = True
            self.delayed_model_parameters = p

    def _is_vector(self,d):
        sz = d.size()
        if len(sz)==1:
            return True
        else:
            return False

    def _is_tensor(self,d):
        sz = d.size()
        if len(sz)>1:
            return True
        else:
            return False

    def _aux_do_weight_clipping_norm(self,pars,desired_norm):
        """does weight clipping but only for conv or bias layers (assuming they are named as such); be careful with the namimg here"""
        if self.weight_clipping_value > 0:
            for key in pars:
                # only do the clipping if it is a conv layer or a bias term
                if key.lower().find('conv')>0 or key.lower().find('bias')>0:
                    p = pars[key]
                    if self._is_vector(p.data):
                        # just normalize this vector component-by-component, norm does not matter here as these are only scalars
                        p.data = p.data.clamp_(-self.weight_clipping_value, self.weight_clipping_value)
                    elif self._is_tensor(p.data):
                        # normalize sample-by-sample individually
                        for b in range(p.data.size()[0]):
                            param_norm = p.data[b, ...].norm(desired_norm)
                            if param_norm > self.weight_clipping_value:
                                clip_coef = self.weight_clipping_value / param_norm
                                p.data[b, ...].mul_(clip_coef)
                    else:
                        raise ValueError('Unknown data type; I do not know how to clip this')

    def _do_shared_weight_clipping_pre_lsm(self):
        multi_gaussian_weights = self.params['model']['registration_model']['forward_model']['smoother'][('multi_gaussian_weights', -1, 'the used multi gaussian weights')]
        if multi_gaussian_weights==-1:
            raise ValueError('The multi-gaussian weights should have been set before')
        multi_gaussian_weights = np.array(multi_gaussian_weights)

        sp = self.get_shared_model_parameters()
        for key in sp:
            if key.lower().find('pre_lsm_weights') > 0:
                p = sp[key]
                sz = p.size() #0 dim is weight dimension
                if sz[0]!=len(multi_gaussian_weights):
                    raise ValueError('Number of multi-Gaussian weights needs to be {}, but got {}'.format(sz[0],len(multi_gaussian_weights)))
                for w in range(sz[0]):
                    # this is to assure that the weights are always between 0 and 1 (when using the WeightedLinearSoftmax
                    p[w,...].data.clamp_(0.0-multi_gaussian_weights[w],1.0-multi_gaussian_weights[w])
                
    def _do_individual_weight_clipping_l1(self):
        ip = self.get_individual_model_parameters()
        self._aux_do_weight_clipping_norm(pars=ip,desired_norm=1)

    def _do_shared_weight_clipping_l1(self):
        sp = self.get_shared_model_parameters()
        self._aux_do_weight_clipping_norm(pars=sp,desired_norm=1)

    def _do_individual_weight_clipping_l2(self):
        ip = self.get_individual_model_parameters()
        self._aux_do_weight_clipping_norm(pars=ip, desired_norm=2)

    def _do_shared_weight_clipping_l2(self):
        sp = self.get_shared_model_parameters()
        self._aux_do_weight_clipping_norm(pars=sp, desired_norm=2)

    def _do_weight_clipping(self):
        """performs weight clipping, if desired"""
        if self.weight_clipping_type is not None:
            possible_modes = ['l1', 'l2', 'l1_individual', 'l2_individual', 'l1_shared', 'l2_shared', 'pre_lsm_weights']
            if self.weight_clipping_type in possible_modes:
                if self.weight_clipping_type=='l1':
                    self._do_shared_weight_clipping_l1()
                    self._do_individual_weight_clipping_l1()
                elif self.weight_clipping_type=='l2':
                    self._do_shared_weight_clipping_l2()
                    self._do_individual_weight_clipping_l2()
                elif self.weight_clipping_type=='l1_individual':
                    self._do_individual_weight_clipping_l1()
                elif self.weight_clipping_type=='l2_individual':
                    self._do_individual_weight_clipping_l2()
                elif self.weight_clipping_type=='l1_shared':
                    self._do_shared_weight_clipping_l1()
                elif self.weight_clipping_type=='l2_shared':
                    self._do_shared_weight_clipping_l2()
                elif self.weight_clipping_type=='pre_lsm_weights':
                    self._do_shared_weight_clipping_pre_lsm()
                else:
                    raise ValueError('Illegal weight clipping type: {}'.format(self.weight_clipping_type))
            else:
                raise ValueError('Weight clipping needs to be: [None|l1|l2|l1_individual|l2_individual|l1_shared|l2_shared]')

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters
        """
        return self.model.get_registration_parameters()

    def set_shared_model_parameters(self,p):
        """
        Set only the shared parameters of the model

        :param p: shared registration parameters as an ordered dict
        :return: n/a
        """

        self.model.set_shared_registration_parameters(p)

    def get_shared_model_parameters_and_buffers(self):
        """
        Returns only the model parameters that are shared between models and the shared buffers associated w/ it.

        :return: shared model parameters and buffers
        """
        return self.model.get_shared_registration_parameters_and_buffers()

    def get_shared_model_parameters(self):
        """
        Returns only the model parameters that are shared between models.

        :return: shared model parameters
        """
        return self.model.get_shared_registration_parameters()

    def set_individual_model_parameters(self,p):
        """
        Set only the individual parameters of the model

        :param p: individual registration parameters as an ordered dict
        :return: n/a
        """

        self.model.set_individual_registration_parameters(p)

    def get_individual_model_parameters(self):
        """
        Returns only the model parameters that individual to a model (i.e., not shared).

        :return: individual model parameters
        """
        return self.model.get_individual_registration_parameters()

    def _collect_individual_or_shared_parameters_in_list(self,pars):
        pl = []
        for p_key in pars:
            pl.append(pars[p_key])
        return pl

    def load_shared_state_dict(self,sd):
        """
        Loads the shared part of a state dictionary
        :param sd: shared state dictionary
        :return: n/a
        """
        self.model.load_shared_state_dict(sd)

    def shared_state_dict(self):
        """
        Returns the shared part of a state dictionary
        :return:
        """
        return self.model.shared_state_dict()

    def load_individual_state_dict(self):
        raise ValueError('Not yet implemented')

    def individual_state_dict(self):
        raise ValueError('Not yet implemented')

    def upsample_model_parameters(self, desiredSize):
        """
        Upsamples the model parameters

        :param desiredSize: desired size after upsampling, e.g., [100,20,50]
        :return: returns a tuple (upsampled_parameters,upsampled_spacing)
        """
        return self.model.upsample_registration_parameters(desiredSize)

    def downsample_model_parameters(self, desiredSize):
        """
        Downsamples the model parameters

        :param desiredSize: desired size after downsampling, e.g., [50,50,40]
        :return: returns a tuple (downsampled_parameters,downsampled_spacing)
        """
        return self.model.downsample_registration_parameters(desiredSize)

    def _set_number_of_iterations_from_multi_scale(self, nrIter):
        """
        Same as set_number_of_iterations with the exception that this is not recored in the parameter structure since it comes from the multi-scale setting
        :param nrIter: number of iterations
        """
        self.nrOfIterations = nrIter

    def set_number_of_iterations(self, nrIter):
        """
        Set the number of iterations of the optimizer

        :param nrIter: number of iterations
        """
        self.params['optimizer'][('single_scale', {}, 'single scale settings')]
        self.params['optimizer']['single_scale']['nr_of_iterations'] = (nrIter, 'number of iterations')

        self.nrOfIterations = nrIter

    def get_number_of_iterations(self):
        """
        Returns the number of iterations of the solver

        :return: number of set iterations
        """
        return self.nrOfIterations

    def _closure(self):
        self.optimizer_instance.zero_grad()
        # 1) Forward pass: Compute predicted y by passing x to the model
        # 2) Compute loss

        if self.iter_count <0:#  or self.iter_count==0:
            # first define variables that will be passed to the model and the criterion (for further use)

            over_scale_iter_count = self.iter_count if self.over_scale_iter_count is None else self.over_scale_iter_count + self.iter_count
            opt_variables = {'iter': self.iter_count, 'epoch': self.current_epoch, 'scale': self.n_scale,
                            'over_scale_iter_count': over_scale_iter_count}

            self.rec_IWarped, self.rec_phiWarped, self.rec_phiInverseWarped = model_evaluation.evaluate_model_low_level_interface(
                model=self.model,
                I_source=self.ISource,
                opt_variables=opt_variables,
                use_map=self.useMap,
                initial_map=self.initialMap,
                compute_inverse_map=self.compute_inverse_map,
                initial_inverse_map=self.initialInverseMap,
                map_low_res_factor=self.mapLowResFactor,
                sampler=self.sampler,
                low_res_spacing=self.lowResSpacing,
                spline_order=self.spline_order,
                low_res_I_source=self.lowResISource,
                low_res_initial_map=self.lowResInitialMap,
                low_res_initial_inverse_map=self.lowResInitialInverseMap,
                compute_similarity_measure_at_low_res=self.compute_similarity_measure_at_low_res)

            # compute the respective losses
            if self.useMap:
                if self.mapLowResFactor is not None and self.compute_similarity_measure_at_low_res:
                    loss_overall_energy, sim_energy, reg_energy = self.criterion(self.lowResInitialMap, self.rec_phiWarped,
                                                                                self.lowResISource, self.lowResITarget,
                                                                                self.lowResISource,
                                                                                self.model.get_variables_to_transfer_to_loss_function(),
                                                                                opt_variables)
                else:
                    loss_overall_energy,sim_energy,reg_energy = self.criterion(self.initialMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource,
                                                                            self.model.get_variables_to_transfer_to_loss_function(),
                                                                            opt_variables)
            else:
                loss_overall_energy,sim_energy,reg_energy = self.criterion(self.rec_IWarped, self.ISource, self.ITarget,
                                    self.model.get_variables_to_transfer_to_loss_function(),
                                    opt_variables )

            # to support consensus optimization we have the option of adding a penalty term
            # based on shared parameters
            opt_par_loss_energy = self.compute_optimizer_parameter_loss(self.model.get_shared_registration_parameters())

            # reconstruction loss
            # proj, idx = self.projection(self.ITarget)
            recon_loss_energy = torch.tensor([0.]).to(torch.device('cuda'))
            recon_reg_energy = torch.tensor([0.]).to(torch.device('cuda'))
            # # recon_loss_energy = recon_loss_energy + F.l1_loss(self.ITargetProj[:,idx], proj, reduce="mean") / (self.recon_sigma ** 2)
            # recon_loss_energy = recon_loss_energy + ((self.ITargetProj[:,idx]-proj)**2).sum() / (self.recon_sigma ** 2) / self.recon_volumn
            # recon_reg_energy = torch.sum(F.relu(-1*self.ITarget)) #+ torch.mean(image_gradient(self.ITarget))

            #reg_rec joint loss
            I1Warped = utils.compute_warped_image_multiNC(self.ISource,
                                                                        self.rec_phiWarped,
                                                                        self.spacing,
                                                                        self.spline_order,
                                                                        zero_boundary=False)
            proj_reg = self.projection(I1Warped)
            reg_rec_joint_loss_energy = torch.tensor([0.]).to(torch.device('cuda'))
            # reg_rec_joint_loss_energy = reg_rec_joint_loss_energy + ((self.ITargetProj[:,idx]-proj_reg)**2).sum() / (self.recon_sigma ** 2) / self.recon_volumn


            loss_overall_energy  = loss_overall_energy + opt_par_loss_energy + reg_rec_joint_loss_energy #recon_loss_energy + recon_reg_energy + reg_rec_joint_loss_energy
            loss_overall_energy.backward()

            # do gradient clipping
            if self.clip_individual_gradient:
                current_individual_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._collect_individual_or_shared_parameters_in_list(self.get_individual_model_parameters()),
                    self.clip_individual_gradient_value)

                if self.clip_display:
                    if current_individual_grad_norm>self.clip_individual_gradient_value:
                        print('INFO: Individual gradient was clipped: {} -> {}'.format(current_individual_grad_norm,self.clip_individual_gradient_value))

            if self.clip_shared_gradient:
                current_shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._collect_individual_or_shared_parameters_in_list(self.get_shared_model_parameters()),
                    self.clip_shared_gradient_value)

                if self.clip_display:
                    if current_shared_grad_norm > self.clip_shared_gradient_value:
                        print('INFO: Shared gradient was clipped: {} -> {}'.format(current_shared_grad_norm,
                                                                                    self.clip_shared_gradient_value))

            self.rec_custom_optimizer_output_string = self.model.get_custom_optimizer_output_string()
            self.rec_custom_optimizer_output_values = self.model.get_custom_optimizer_output_values()

            self.rec_energy = loss_overall_energy
            self.rec_similarityEnergy = sim_energy
            self.rec_regEnergy = reg_energy
            self.rec_opt_par_loss_energy = opt_par_loss_energy
            self.rec_recon_energy = recon_loss_energy
            self.rec_recon_reg_energy = recon_reg_energy
            self.reg_rec_joint_loss_energy = reg_rec_joint_loss_energy
        else:
            # # reconstruction loss
            # proj, idx = self.projection(self.ITarget)
            # recon_loss_energy = torch.tensor([0.]).to(torch.device('cuda'))
            # # recon_loss_energy = recon_loss_energy + F.l1_loss(self.ITargetProj[:,idx], proj, reduce="mean") / (self.recon_sigma ** 2)
            # recon_loss_energy = recon_loss_energy + ((self.ITargetProj[:,idx]-proj)**2).sum() / (self.recon_sigma ** 2) / self.recon_volumn
            # recon_reg_energy = torch.sum(F.relu(-1*self.ITarget)) #+ torch.mean(image_gradient(self.ITarget))

            # #reg_rec joint loss
            # # I1Warped = utils.compute_warped_image_multiNC(self.ISource,
            # #                                                             self.rec_phiWarped,
            # #                                                             self.spacing,
            # #                                                             self.spline_order,
            # #                                                             zero_boundary=False)
            # # proj_reg, idx = self.projection(I1Warped)
            # reg_rec_joint_loss_energy = torch.tensor([0.]).to(torch.device('cuda'))
            # # reg_rec_joint_loss_energy = reg_rec_joint_loss_energy + ((self.ITargetProj[:,idx]-proj_reg)**2).sum() / (self.recon_sigma ** 2) / self.recon_volumn


            # loss_overall_energy  = recon_loss_energy + recon_reg_energy #+ reg_rec_joint_loss_energy
            # loss_overall_energy.backward()

            # self.rec_recon_energy = recon_loss_energy
            # self.rec_recon_reg_energy = recon_reg_energy
            # self.reg_rec_joint_loss_energy = reg_rec_joint_loss_energy

            # first define variables that will be passed to the model and the criterion (for further use)

            over_scale_iter_count = self.iter_count if self.over_scale_iter_count is None else self.over_scale_iter_count + self.iter_count
            opt_variables = {'iter': self.iter_count, 'epoch': self.current_epoch, 'scale': self.n_scale,
                            'over_scale_iter_count': over_scale_iter_count}

            self.rec_IWarped, self.rec_phiWarped, self.rec_phiInverseWarped = model_evaluation.evaluate_model_low_level_interface(
                model=self.model,
                I_source=self.ISource,
                opt_variables=opt_variables,
                use_map=self.useMap,
                initial_map=self.initialMap,
                compute_inverse_map=self.compute_inverse_map,
                initial_inverse_map=self.initialInverseMap,
                map_low_res_factor=self.mapLowResFactor,
                sampler=self.sampler,
                low_res_spacing=self.lowResSpacing,
                spline_order=self.spline_order,
                low_res_I_source=self.lowResISource,
                low_res_initial_map=self.lowResInitialMap,
                low_res_initial_inverse_map=self.lowResInitialInverseMap,
                compute_similarity_measure_at_low_res=self.compute_similarity_measure_at_low_res)

            # compute the respective losses
            if self.useMap:
                if self.mapLowResFactor is not None and self.compute_similarity_measure_at_low_res:
                    loss_overall_energy, sim_energy, reg_energy = self.criterion(self.lowResInitialMap, self.rec_phiWarped,
                                                                                self.lowResISource, self.lowResITarget,
                                                                                self.lowResISource,
                                                                                self.model.get_variables_to_transfer_to_loss_function(),
                                                                                opt_variables)
                else:
                    loss_overall_energy,sim_energy,reg_energy = self.criterion(self.initialMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource,
                                                                            self.model.get_variables_to_transfer_to_loss_function(),
                                                                            opt_variables)
            else:
                loss_overall_energy,sim_energy,reg_energy = self.criterion(self.rec_IWarped, self.ISource, self.ITarget,
                                    self.model.get_variables_to_transfer_to_loss_function(),
                                    opt_variables )

            # to support consensus optimization we have the option of adding a penalty term
            # based on shared parameters
            opt_par_loss_energy = self.compute_optimizer_parameter_loss(self.model.get_shared_registration_parameters())

            # reconstruction loss
            proj = self.projection(self.ITarget)
            recon_loss_energy = torch.tensor([0.]).to(torch.device('cuda'))
            
            # recon_loss_energy = recon_loss_energy + F.l1_loss(self.ITargetProj[:,idx], proj, reduce="mean") / (self.recon_sigma ** 2)
            # recon_loss_energy = recon_loss_energy + ((self.ITargetProj[:,idx]-proj)**2).sum() / (self.recon_sigma ** 2) / self.recon_volumn
            recon_loss_energy = recon_loss_energy + self.proj_sim_measure.compute_similarity(self.ITargetProj, proj)/proj.shape[1]
            recon_reg_energy = torch.sum(F.relu(-1*self.ITarget)) + torch.mean(image_gradient(self.ITarget))

            #reg_rec joint loss
            I1Warped = utils.compute_warped_image_multiNC(self.ISource,
                                                                        self.rec_phiWarped,
                                                                        self.spacing,
                                                                        self.spline_order,
                                                                        zero_boundary=False)
            proj_reg = self.projection(I1Warped)
            reg_rec_joint_loss_energy = torch.tensor([0.]).to(torch.device('cuda'))
            reg_rec_joint_loss_energy = reg_rec_joint_loss_energy + self.proj_sim_measure.compute_similarity(self.ITargetProj, proj_reg)/proj_reg.shape[1]
            # reg_rec_joint_loss_energy = reg_rec_joint_loss_energy + ((self.ITargetProj[:,idx]-proj_reg)**2).sum() / (self.recon_sigma ** 2) / self.recon_volumn

            # loss_overall_energy  = reg_energy + opt_par_loss_energy + recon_loss_energy + recon_reg_energy + reg_rec_joint_loss_energy
            loss_overall_energy  = loss_overall_energy + opt_par_loss_energy + recon_loss_energy + recon_reg_energy + reg_rec_joint_loss_energy
            loss_overall_energy.backward()

            # do gradient clipping
            if self.clip_individual_gradient:
                current_individual_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._collect_individual_or_shared_parameters_in_list(self.get_individual_model_parameters()),
                    self.clip_individual_gradient_value)

                if self.clip_display:
                    if current_individual_grad_norm>self.clip_individual_gradient_value:
                        print('INFO: Individual gradient was clipped: {} -> {}'.format(current_individual_grad_norm,self.clip_individual_gradient_value))

            if self.clip_shared_gradient:
                current_shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._collect_individual_or_shared_parameters_in_list(self.get_shared_model_parameters()),
                    self.clip_shared_gradient_value)

                if self.clip_display:
                    if current_shared_grad_norm > self.clip_shared_gradient_value:
                        print('INFO: Shared gradient was clipped: {} -> {}'.format(current_shared_grad_norm,
                                                                                    self.clip_shared_gradient_value))

            self.rec_custom_optimizer_output_string = self.model.get_custom_optimizer_output_string()
            self.rec_custom_optimizer_output_values = self.model.get_custom_optimizer_output_values()

            self.rec_energy = loss_overall_energy
            self.rec_similarityEnergy = sim_energy
            self.rec_regEnergy = reg_energy
            self.rec_opt_par_loss_energy = opt_par_loss_energy
            self.rec_recon_energy = recon_loss_energy
            self.rec_recon_reg_energy = recon_reg_energy
            self.reg_rec_joint_loss_energy = reg_rec_joint_loss_energy

        # if self.useMap:
        #
        #    if self.iter_count % 1 == 0:
        #        self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
        #            self.identityMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource, self.model.get_variables_to_transfer_to_loss_function())
        # else:
        #    if self.iter_count % 1 == 0:
        #        self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
        #            self.rec_IWarped, self.ISource, self.ITarget, self.model.get_variables_to_transfer_to_loss_function())

        return loss_overall_energy

    def analysis(self, energy, similarityEnergy, regEnergy, opt_par_energy, rec_Energy, rec_recon_reg_energy, reg_rec_joint_loss_energy, phi_or_warped_image, custom_optimizer_output_string ='', custom_optimizer_output_values=None, force_visualization=False):
        """
        print out the and visualize the result
        :param energy:
        :param similarityEnergy:
        :param regEnergy:
        :param opt_par_energy
        :param phi_or_warped_image:
        :return: returns tuple: first entry True if termination tolerance was reached, otherwise returns False; second entry if the image was visualized
        """

        current_batch_size = phi_or_warped_image.size()[0]

        was_visualized = False
        reached_tolerance = False

        cur_energy = utils.t2np(energy.float())
        # energy analysis

        self._add_to_history('iter', self.iter_count)
        self._add_to_history('energy', cur_energy[0])
        self._add_to_history('similarity_energy', utils.t2np(similarityEnergy.float())[0])
        self._add_to_history('regularization_energy', utils.t2np(regEnergy.float()))
        self._add_to_history('opt_par_energy', utils.t2np(opt_par_energy.float())[0])
        self._add_to_history('reconstruction_energy', utils.t2np(rec_Energy.float()))

        if custom_optimizer_output_values is not None:
            for key in custom_optimizer_output_values:
                self._add_to_history(key,custom_optimizer_output_values[key])

        if self.last_energy is not None:

            # relative function tolerance: |f(xi)-f(xi+1)|/(1+|f(xi)|)
            self.rel_f = abs(self.last_energy - cur_energy) / (1 + abs(cur_energy))
            self._add_to_history('relF', self.rel_f[0])

            if self.show_iteration_output:
                cprint('{iter:5d}-Tot: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | optParE={optParE:08.4f} | reconE={reconE:08.4f} | reconRegE={reconRegE:08.4f} | regRecJointE={regRecJointE:08.4f} | relF={relF:08.4f} | {cos}'
                       .format(iter=self.iter_count,
                               energy=utils.get_scalar(cur_energy),
                               similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())),
                               regE=utils.get_scalar(utils.t2np(regEnergy.float())),
                               optParE=utils.get_scalar(utils.t2np(opt_par_energy.float())),
                               reconE=utils.get_scalar(utils.t2np(rec_Energy.float())),
                               reconRegE=utils.get_scalar(utils.t2np(rec_recon_reg_energy.float())),
                               regRecJointE=utils.get_scalar(utils.t2np(reg_rec_joint_loss_energy.float())),
                               relF=utils.get_scalar(self.rel_f),
                               cos=custom_optimizer_output_string), 'red')
                cprint('{iter:5d}-Img: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | reconE={reconE:08.4f} | reconRegE={reconRegE:08.4f} | regRecJointE={regRecJointE:08.4f}'
                       .format(iter=self.iter_count,
                               energy=utils.get_scalar(cur_energy) / current_batch_size,
                               similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())) / current_batch_size,
                               regE=utils.get_scalar(utils.t2np(regEnergy.float())) / current_batch_size,
                               reconE=utils.get_scalar(utils.t2np(rec_Energy.float())),
                               reconRegE=utils.get_scalar(utils.t2np(rec_recon_reg_energy.float())),
                               regRecJointE=utils.get_scalar(utils.t2np(reg_rec_joint_loss_energy.float()))), 'blue')

            # check if relative convergence tolerance is reached
            if self.rel_f < self.rel_ftol:
                if self.show_iteration_output:
                    print('Reached relative function tolerance of = ' + str(self.rel_ftol))
                reached_tolerance = True

        else:
            self._add_to_history('relF', None)
            if self.show_iteration_output:
                cprint('{iter:5d}-Tot: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | optParE={optParE:08.4f} | reconE={reconE:08.4f} | reconRegE={reconRegE:08.4f} | regRecJointE={regRecJointE:08.4f} | relF=  n/a    | {cos}'
                      .format(iter=self.iter_count,
                              energy=utils.get_scalar(cur_energy),
                              similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())),
                              regE=utils.get_scalar(utils.t2np(regEnergy.float())),
                              optParE=utils.get_scalar(utils.t2np(opt_par_energy.float())),
                              reconE=utils.get_scalar(utils.t2np(rec_Energy.float())),
                              reconRegE=utils.get_scalar(utils.t2np(rec_recon_reg_energy.float())),
                              regRecJointE=utils.get_scalar(utils.t2np(reg_rec_joint_loss_energy.float())),
                              cos=custom_optimizer_output_string), 'red')
                cprint('{iter:5d}-Img: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | reconE={reconE:08.4f} | reconRegE={reconRegE:08.4f} | regRecJointE={regRecJointE:08.4f}'
                      .format(iter=self.iter_count,
                              energy=utils.get_scalar(cur_energy)/current_batch_size,
                              similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float()))/current_batch_size,
                              regE=utils.get_scalar(utils.t2np(regEnergy.float()))/current_batch_size,
                              reconE=utils.get_scalar(utils.t2np(rec_Energy.float())),
                              reconRegE=utils.get_scalar(utils.t2np(rec_recon_reg_energy.float())),
                              regRecJointE=utils.get_scalar(utils.t2np(reg_rec_joint_loss_energy.float()))),'blue')

        iter_count = self.iter_count
        self.last_energy = cur_energy

        if self.recording_step is not None:
            if iter_count % self.recording_step == 0 or iter_count == 0:
                if self.useMap:
                    if self.compute_similarity_measure_at_low_res:
                        I1Warped = utils.compute_warped_image_multiNC(self.lowResISource,
                                                                      phi_or_warped_image,
                                                                      self.lowResSpacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        lowResLWarped = utils.get_warped_label_map(self.lowResLSource,
                                                                   phi_or_warped_image,
                                                                   self.spacing)
                        self.history['recording'].append({
                            'iter': iter_count,
                            'iS': utils.t2np(self.ISource),
                            'iT': utils.t2np(self.ITarget),
                            'iW': utils.t2np(I1Warped),
                            'iSL': utils.t2np(self.lowResLSource) if self.lowResLSource is not None else None,
                            'iTL': utils.t2np(self.lowResLTarget) if self.lowResLTarget is not None else None,
                            'iWL': utils.t2np(lowResLWarped) if self.lowResLWarped is not None else None,
                            'phiWarped': utils.t2np(phi_or_warped_image)
                        })
                    else:
                        I1Warped = utils.compute_warped_image_multiNC(self.ISource,
                                                                      phi_or_warped_image,
                                                                      self.spacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        LWarped = None
                        if self.LSource is not None and self.LTarget is not None:
                            LWarped = utils.get_warped_label_map(self.LSource,
                                                                 phi_or_warped_image,
                                                                 self.spacing)
                        self.history['recording'].append({
                            'iter': iter_count,
                            'iS': utils.t2np(self.ISource),
                            'iT': utils.t2np(self.ITarget),
                            'iW': utils.t2np(I1Warped),
                            'iSL': utils.t2np(self.LSource) if self.LSource is not None else None,
                            'iTL': utils.t2np(self.LTarget) if self.LTarget is not None else None,
                            'iWL': utils.t2np(LWarped) if LWarped is not None else None,
                            'phiWarped': utils.t2np(phi_or_warped_image)
                        })
                else:
                    self.history['recording'].append({
                        'iter': iter_count,
                        'iS': utils.t2np(self.ISource),
                        'iT': utils.t2np(self.ITarget),
                        'iW': utils.t2np(phi_or_warped_image)
                    })

        if self.visualize or self.save_fig:
            visual_param = {}
            visual_param['visualize'] = self.visualize
            visual_param['save_fig'] = self.save_fig
            visual_param['save_fig_num'] = self.save_fig_num
            if self.save_fig:
                visual_param['save_fig_path'] = self.save_fig_path
                visual_param['save_fig_path_byname'] = os.path.join(self.save_fig_path, 'byname')
                visual_param['save_fig_path_byiter'] = os.path.join(self.save_fig_path, 'byiter')
                visual_param['pair_name'] = self.pair_name
                visual_param['iter'] = 'scale_'+str(self.n_scale) + '_iter_' + str(self.iter_count)

            if self.visualize_step and (iter_count % self.visualize_step == 0) or (iter_count == self.nrOfIterations-1) or force_visualization:
                was_visualized = True
                if self.useMap and self.mapLowResFactor is not None:
                    vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize(self.lowResISource)
                else:
                    vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize(self.ISource)

                if self.useMap:
                    if self.compute_similarity_measure_at_low_res:
                        I1Warped = utils.compute_warped_image_multiNC(self.lowResISource,
                                                                      phi_or_warped_image,
                                                                      self.lowResSpacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        lowResLWarped = utils.get_warped_label_map(self.lowResLSource,
                                                                   phi_or_warped_image,
                                                                   self.spacing)
                        vizReg.show_current_images(iter=iter_count,
                                                   iS=self.lowResISource,
                                                   iT=self.lowResITarget,
                                                   iW=I1Warped,
                                                   iSL=self.lowResLSource,
                                                   iTL=self.lowResLTarget,
                                                   iWL=lowResLWarped,
                                                   vizImages=vizImage,
                                                   vizName=vizName,
                                                   phiWarped=phi_or_warped_image,
                                                   visual_param=visual_param)

                    else:
                        I1Warped = utils.compute_warped_image_multiNC(self.ISource,
                                                                      phi_or_warped_image,
                                                                      self.spacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        vizImage = vizImage if len(vizImage)>2 else None
                        LWarped = None
                        if self.LSource is not None  and self.LTarget is not None:
                            LWarped = utils.get_warped_label_map(self.LSource,
                                                                 phi_or_warped_image,
                                                                 self.spacing)

                        #################################################
                        ## Modify so that mermaid can show 3d 2d log
                        if self.params["model"]["registration_model"]["similarity_measure"]["type"] == "projection":
                            emi_pose = self.params["model"]['registration_model']['emitter_pos_scale_list'][1]
                            sample_rate = self.params["model"]['registration_model']['sample_rate']
                            resolution = [int(self.ITarget.shape[2]), int(self.ITarget.shape[3])]
                            ISource = self.criterion.similarityMeasure.project(self.ISource, np.array(emi_pose), resolution, sample_rate)
                            I1Warped = self.criterion.similarityMeasure.project(I1Warped, np.array(emi_pose), resolution, sample_rate)
                            ITarget = self.ITarget[:, 1, :, :].unsqueeze(0)
                            visual_param["prefix"] = self.params["model"]["registration_model"]["type"]
                            phi = None
                        else:
                            ISource = self.ISource
                            ITarget = self.ITarget
                            phi = self.rec_phiWarped
                        ##################################################

                        vizReg.show_current_images(iter=iter_count,
                                                   iS=ISource,
                                                   iT=ITarget,
                                                   iW=I1Warped,
                                                   iSL=self.LSource,
                                                   iTL=self.LTarget,
                                                   iWL=LWarped,
                                                   vizImages=vizImage,
                                                   vizName=vizName,
                                                   phiWarped=phi,
                                                   visual_param=visual_param)
                else:
                    vizReg.show_current_images(iter=iter_count,
                                               iS=self.ISource,
                                               iT=self.ITarget,
                                               iW=phi_or_warped_image,
                                               vizImages=vizImage,
                                               vizName=vizName,
                                               phiWarped=None,
                                               visual_param=visual_param)

        return reached_tolerance, was_visualized

    def _debugging_saving_intermid_img(self,img=None,is_label_map=False, append=''):
        folder_path = os.path.join(self.save_fig_path,'debugging')
        folder_path = os.path.join(folder_path, self.pair_name[0])
        make_dir(folder_path)
        file_name = 'scale_'+str(self.n_scale) + '_iter_' + str(self.iter_count)+append
        file_name=file_name.replace('.','_')
        if is_label_map:
            file_name += '_label'
        path = os.path.join(folder_path,file_name+'.nii.gz')
        im_io = FIO.ImageIO()
        im_io.write(path, np.squeeze(img.detach().cpu().numpy()))

    # todo: write these parameter/optimizer functions also for shared parameters and all parameters
    def set_sgd_shared_model_parameters_and_optimizer_states(self, pars):
        """
               Set the individual model parameters and states that may be stored by the optimizer such as the momentum.
               Expects as input what get_sgd_individual_model_parameters_and_optimizer_states creates as output,
               but potentially multiple copies of it (as generated by a pyTorch dataloader). I.e., it takes in a dataloader sample.
               NOTE: currently only supports SGD

               :param pars: parameter list as produced by get_sgd_individual_model_parameters_and_optimizer_states
               :return: n/a
               """
        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        if len(pars) == 0:
            print('WARNING: found no values')
            return

        # the optimizer (if properly initialized) already holds pointers to the model parameters and the optimizer states
        # so we can set everything in one swoop here

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        # this input will represent a sample from a pytorch dataloader

        # wrap the parameters in a list if needed (so we can mirror the setup from get_sgd_...
        if type(pars) == list:
            use_pars = pars
        else:
            use_pars = [pars]

        for p in use_pars:
            if 'is_shared' in p:
                if p['is_shared']:
                    current_name = p['name']

                    assert (torch.is_tensor(p['model_params']))
                    current_model_params = p['model_params']

                    if 'momentum_buffer' in p:
                        assert (torch.is_tensor(p['momentum_buffer']))
                        current_momentum_buffer = p['momentum_buffer']
                    else:
                        current_momentum_buffer = None

                    # now we need to match this with the parameters and the state of the SGD optimizer
                    model_par = self._sgd_name_to_model_par[current_name]
                    model_par.data.copy_(current_model_params)

                    # and now do the same with the state
                    param_state = self.optimizer_instance.state[model_par]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].copy_(current_momentum_buffer)

    def set_sgd_individual_model_parameters_and_optimizer_states(self, pars):
        """
        Set the individual model parameters and states that may be stored by the optimizer such as the momentum.
        Expects as input what get_sgd_individual_model_parameters_and_optimizer_states creates as output,
        but potentially multiple copies of it (as generated by a pyTorch dataloader). I.e., it takes in a dataloader sample.
        NOTE: currently only supports SGD

        :param pars: parameter list as produced by get_sgd_individual_model_parameters_and_optimizer_states
        :return: n/a
        """
        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        if len(pars) == 0:
            print('WARNING: found no values')
            return

        # the optimizer (if properly initialized) already holds pointers to the model parameters and the optimizer states
        # so we can set everything in one swoop here

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        # this input will represent a sample from a pytorch dataloader

        # wrap the parameters in a list if needed (so we can mirror the setup from get_sgd_...
        if type(pars)==list:
            use_pars = pars
        else:
            use_pars = [pars]

        for p in use_pars:
            if 'is_shared' in p:
                if not p['is_shared'][0]: # need to grab the first one, because the dataloader replicated these entries
                    current_name = p['name'][0]

                    assert( torch.is_tensor(p['model_params']))
                    current_model_params = p['model_params']

                    if 'momentum_buffer' in p:
                        assert( torch.is_tensor(p['momentum_buffer']) )
                        current_momentum_buffer = p['momentum_buffer']
                    else:
                        current_momentum_buffer = None

                    # now we need to match this with the parameters and the state of the SGD optimizer
                    model_par = self._sgd_name_to_model_par[current_name]
                    model_par.data.copy_(current_model_params)

                    # and now do the same with the state
                    param_state = self.optimizer_instance.state[model_par]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].copy_(current_momentum_buffer)

    def _convert_obj_with_parameters_to_obj_with_tensors(self, p):
        """
        Converts structures that consist of lists and dictionaries with parameters to tensors

        :param p: parameter structure
        :return: object with parameters converted to tensors
        """

        if type(p) == list:
            ret_p = []
            for e in p:
                ret_p.append(self._convert_obj_with_parameters_to_obj_with_tensors(e))
            return ret_p
        elif type(p) == dict:
            ret_p = dict()
            for key in p:
                ret_p[key] = self._convert_obj_with_parameters_to_obj_with_tensors((p[key]))
            return ret_p
        elif type(p) == torch.nn.parameter.Parameter:
            return p.data
        else:
            return p

    def get_sgd_shared_model_parameters(self):
        """
        Gets the model parameters that are shared.

        :return:
        """

        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        d = []

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        for group in self.optimizer_instance.param_groups:

            group_dict = dict()
            group_dict['params'] = []

            for p in group['params']:
                current_group_params = dict()
                # let's first see if this is a shared state
                if self._sgd_par_names[p]['is_shared']:
                    # keep track of the names so we can and batch, so we can read it back in
                    current_group_params.update(self._sgd_par_names[p])
                    # now deal with the optimizer state if available
                    current_group_params['model_params'] = self._convert_obj_with_parameters_to_obj_with_tensors(p)

                    group_dict['params'].append(current_group_params)

            d.append(group_dict)

        return d


    def get_sgd_individual_model_parameters_and_optimizer_states(self):
        """
        Gets the individual model parameters and states that may be stored by the optimizer such as the momentum.
        NOTE: currently only supports SGD

        :return:
        """
        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        d = []

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        for group in self.optimizer_instance.param_groups:

            group_dict = dict()
            group_dict['weight_decay'] = group['weight_decay']
            group_dict['momentum'] = group['momentum']
            group_dict['dampening'] = group['dampening']
            group_dict['nesterov'] = group['nesterov']
            group_dict['lr'] = group['lr']

            group_dict['params'] = []

            for p in group['params']:
                current_group_params = dict()
                # let's first see if this is a shared state
                if not self._sgd_par_names[p]['is_shared']:
                    # keep track of the names so we can and batch, so we can read it back in
                    current_group_params.update(self._sgd_par_names[p])
                    # now deal with the optimizer state if available
                    current_group_params['model_params'] = self._convert_obj_with_parameters_to_obj_with_tensors(p)
                    if group['momentum'] != 0:
                        param_state = self.optimizer_instance.state[p]
                        if 'momentum_buffer' in param_state:
                            current_group_params['momentum_buffer'] = self._convert_obj_with_parameters_to_obj_with_tensors(param_state['momentum_buffer'])

                    group_dict['params'].append(current_group_params)

            d.append(group_dict)

        return d

    def _remove_state_variables_for_individual_parameters(self,individual_pars):
        """
        Removes the optimizer state for individual parameters.
        This is required at the beginning as we do not want to reuse the SGD momentum for example for an unrelated registration.

        :param individual_pars: individual parameters are returned by get_sgd_individual_model_parameters_and_optimizer_states
        :return: n/a
        """

        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        for group in self.optimizer_instance.param_groups:

            for p in group['params']:
                # let's first see if this is a shared state
                if not self._sgd_par_names[p]['is_shared']:
                    # we want to delete the state of this one
                    self.optimizer_instance.state.pop(p)


    def _create_optimizer_parameter_dictionary(self,individual_pars, shared_pars,
                                              settings_individual=dict(), settings_shared=dict()):

        par_list = []
        """List of parameters that can directly be passed to an optimizer; different list elements define different parameter groups"""
        par_names = dict()
        """dictionary which maps from a parameters id (i.e., memory) to its description: name/is_shared"""
        # name is the name of the variable
        # is_shared keeps track of if a parameter was declared shared (opposed to individual, which we need for registrations)

        names_to_par = dict()
        """dictionary which maps from a parameter name back to the parameter"""

        # Lin: add the reconstruction parameters
        cd = {'params': self.ITarget}
        recon_settings = settings_individual.copy()
        recon_settings["lr"] = self.recon_lr
        cd.update(recon_settings)
        par_list.append(cd)

        # first deal with the individual parameters
        pl_ind, par_to_name_ind = utils.get_parameter_list_and_par_to_name_dict_from_parameter_dict(individual_pars)
        #cd = {'params': pl_ind}
        cd = {'params': [p for p in pl_ind if p.requires_grad]}
        cd.update(settings_individual)
        par_list.append(cd)
        # add all the names
        for current_par, key in zip(pl_ind, par_to_name_ind):
            par_names[key] = {'name': par_to_name_ind[key], 'is_shared': False}
            names_to_par[par_to_name_ind[key]] = current_par

        # now deal with the shared parameters
        pl_shared, par_to_name_shared = utils.get_parameter_list_and_par_to_name_dict_from_parameter_dict(shared_pars)
        #cd = {'params': pl_shared}
        cd = {'params': [p for p in pl_shared if p.requires_grad]}
        cd.update(settings_shared)
        par_list.append(cd)
        for current_par, key in zip(pl_shared, par_to_name_shared):
            par_names[key] = {'name': par_to_name_shared[key], 'is_shared': True}
            names_to_par[par_to_name_shared[key]] = current_par

        return par_list, par_names, names_to_par

    def _write_out_shared_parameters(self, model_pars, filename):

        # just write out the ones that are shared
        for group in model_pars:
            if 'params' in group:
                was_shared_group = False  # there can only be one
                # create lists that will hold the information for the different batches
                cur_pars = []

                # now iterate through the current parameter list
                for p in group['params']:
                    needs_to_be_saved = True
                    if 'is_shared' in p:
                        if not p['is_shared']:
                            needs_to_be_saved = False

                    if needs_to_be_saved:
                        # we found a shared entry
                        was_shared_group = True
                        cur_pars.append(p)

                # now we have the parameter list for one of the elements of the batch and we can write it out
                if was_shared_group:  # otherwise will be overwritten by a later parameter group
                    torch.save(cur_pars, filename)


    def _write_out_individual_parameters(self, model_pars, filenames):

        batch_size = len(filenames)

        # just write out the ones that are individual
        for group in model_pars:
            if 'params' in group:
                was_individual_group = False  # there can only be one
                # create lists that will hold the information for the different batches
                for b in range(batch_size):
                    cur_pars = []

                    # now iterate through the current parameter list
                    for p in group['params']:
                        if 'is_shared' in p:
                            # we found an individual entry
                            if not p['is_shared']:
                                was_individual_group = True
                                # now go through this dictionary, extract the current batch info in it,
                                # and append it to the current batch parameter list
                                cur_dict = dict()
                                for p_el in p:
                                    if p_el == 'name':
                                        cur_dict['name'] = p[p_el]
                                    elif p_el == 'is_shared':
                                        cur_dict['is_shared'] = p[p_el]
                                    else:
                                        # this will be a tensor so we need to extract the information for the current batch
                                        cur_dict[p_el] = p[p_el][b, ...]

                                cur_pars.append(cur_dict)

                    # now we have the parameter list for one of the elements of the batch and we can write it out
                    if was_individual_group:  # otherwise will be overwritten by a later parameter group
                        torch.save(cur_pars, filenames[b])

    def _get_optimizer_instance(self):

        if (self.model is None) or (self.criterion is None):
            raise ValueError('Please specify a model to solve with set_model first')

        # first check if an optimizer was specified externally

        if self.optimizer is not None:
            # simply instantiate it
            if self.optimizer_name is not None:
                print('Warning: optimizer name = ' + str(self.optimizer_name) +
                      ' specified, but ignored since optimizer was set explicitly')
            opt_instance = self.optimizer(self.model.parameters(), **self.optimizer_params)
            return opt_instance
        else:
            # select it by name
            # TODO: Check what the best way to adapt the tolerances is here; tying it to rel_ftol is not really correct
            if self.optimizer_name is None:
                raise ValueError('Need to select an optimizer')
            elif self.optimizer_name == 'lbfgs_ls':
                if self.last_successful_step_size_taken is not None:
                    desired_lr = self.last_successful_step_size_taken
                else:
                    desired_lr = 1.0
                max_iter = self.params['optimizer']['lbfgs'][('max_iter',1,'maximum number of iterations')]
                max_eval = self.params['optimizer']['lbfgs'][('max_eval',5,'maximum number of evaluation')]
                history_size = self.params['optimizer']['lbfgs'][('history_size',5,'Size of the optimizer history')]
                line_search_fn = self.params['optimizer']['lbfgs'][('line_search_fn','backtracking','Type of line search function')]

                opt_instance = CO.LBFGS_LS(self.model.parameters(),
                                           lr=desired_lr, max_iter=max_iter, max_eval=max_eval,
                                           tolerance_grad=self.rel_ftol * 10, tolerance_change=self.rel_ftol,
                                           history_size=history_size, line_search_fn=line_search_fn)
                return opt_instance
            elif self.optimizer_name == 'sgd':
                #if self.last_successful_step_size_taken is not None:
                #    desired_lr = self.last_successful_step_size_taken
                #else:

                if self.default_learning_rate is not None:
                    current_default_learning_rate = self.default_learning_rate
                    self.params['optimizer']['sgd']['individual']['lr'] = current_default_learning_rate
                    self.params['optimizer']['sgd']['shared']['lr'] = current_default_learning_rate

                else:
                    current_default_learning_rate = 0.01

                desired_lr_individual = self.params['optimizer']['sgd']['individual'][('lr',current_default_learning_rate,'desired learning rate')]
                sgd_momentum_individual = self.params['optimizer']['sgd']['individual'][('momentum',0.9,'sgd momentum')]
                sgd_dampening_individual = self.params['optimizer']['sgd']['individual'][('dampening',0.0,'sgd dampening')]
                sgd_weight_decay_individual = self.params['optimizer']['sgd']['individual'][('weight_decay',0.0,'sgd weight decay')]
                sgd_nesterov_individual = self.params['optimizer']['sgd']['individual'][('nesterov',True,'use Nesterove scheme')]

                desired_lr_shared = self.params['optimizer']['sgd']['shared'][('lr', current_default_learning_rate, 'desired learning rate')]
                sgd_momentum_shared = self.params['optimizer']['sgd']['shared'][('momentum', 0.9, 'sgd momentum')]
                sgd_dampening_shared = self.params['optimizer']['sgd']['shared'][('dampening', 0.0, 'sgd dampening')]
                sgd_weight_decay_shared = self.params['optimizer']['sgd']['shared'][('weight_decay', 0.0, 'sgd weight decay')]
                sgd_nesterov_shared = self.params['optimizer']['sgd']['shared'][('nesterov', True, 'use Nesterove scheme')]

                settings_shared = {'momentum': sgd_momentum_shared,
                                   'dampening': sgd_dampening_shared,
                                   'weight_decay': sgd_weight_decay_shared,
                                   'nesterov': sgd_nesterov_shared,
                                   'lr': desired_lr_shared}

                settings_individual = {'momentum': sgd_momentum_individual,
                                   'dampening': sgd_dampening_individual,
                                   'weight_decay': sgd_weight_decay_individual,
                                   'nesterov': sgd_nesterov_individual,
                                   'lr': desired_lr_individual}

                self._sgd_par_list, self._sgd_par_names, self._sgd_name_to_model_par = self._create_optimizer_parameter_dictionary(
                    self.model.get_individual_registration_parameters(),
                    self.model.get_shared_registration_parameters(),
                    settings_individual=settings_individual,
                    settings_shared=settings_shared)

                opt_instance = torch.optim.SGD(self._sgd_par_list)

                return opt_instance
            elif self.optimizer_name == 'adam':
                if self.last_successful_step_size_taken is not None:
                    desired_lr = self.last_successful_step_size_taken
                else:
                    if self.default_learning_rate is not None:
                        current_default_learning_rate = self.default_learning_rate
                        self.params['optimizer']['adam']['lr'] = current_default_learning_rate
                    else:
                        current_default_learning_rate = 0.01
                    desired_lr = self.params['optimizer']['adam'][('lr',current_default_learning_rate,'desired learning rate')]

                adam_betas = self.params['optimizer']['adam'][('betas',[0.9,0.999],'adam betas')]
                adam_eps = self.params['optimizer']['adam'][('eps',self.rel_ftol,'adam eps')]
                adam_weight_decay = self.params['optimizer']['adam'][('weight_decay',0.0,'adam weight decay')]
                opt_instance = torch.optim.Adam(self.model.parameters(), lr=desired_lr,
                                                betas=adam_betas,
                                                eps=adam_eps,
                                                weight_decay=adam_weight_decay)
                return opt_instance
            else:
                raise ValueError('Optimizer = ' + str(self.optimizer_name) + ' not yet supported')

    def _set_all_still_missing_parameters(self):

        if self.optimizer_name is None:
            self.optimizer_name = self.params['optimizer'][('name','lbfgs_ls','Optimizer (lbfgs|adam|sgd)')]

        if self.model is None:
            model_name = self.params['model']['registration_model'][('type', 'lddmm_shooting_map', "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with suffix '_map' or '_image'")]
            self.params['model']['deformation'][('use_map', True, 'use a map for the solution or not True/False' )]
            self.set_model( model_name )

        if self.nrOfIterations is None: # not externally set, so this will not be a multi-scale solution
            self.params['optimizer'][('single_scale', {}, 'single scale settings')]
            self.nrOfIterations = self.params['optimizer']['single_scale'][('nr_of_iterations', 10, 'number of iterations')]

        # get the optimizer
        if self.optimizer_instance is None:
            self.optimizer_instance = self._get_optimizer_instance()

        if USE_CUDA:
            self.model = self.model.cuda()

        self.compute_low_res_image_if_needed()
        self.optimizer_has_been_initialized = True

    def set_scheduler_patience(self,patience):
        self.params['optimizer']['scheduler']['patience'] = patience
        self.scheduler_patience = patience

    def set_scheduler_patience_silent(self,patience):
        self.scheduler_patience = patience

    def get_scheduler_patience(self):
        return self.scheduler_patience

    def _set_use_external_scheduler(self):
        self._use_external_scheduler = True

    def _set_use_internal_scheduler(self):
        self._use_external_scheduler = False

    def _get_use_external_scheduler(self):
        return self._use_external_scheduler

    def _get_dictionary_to_pass_to_integrator(self):
        """
        This is experimental to allow passing additional parameters to integrators/smoothers, etc.

        :return: dictionary
        """

        d = dict()

        if self.mapLowResFactor is not None:
            d['I0'] = self.lowResISource
            d['I1'] = self.lowResITarget
        else:
            d['I0'] = self.ISource
            d['I1'] = self.ITarget

        return d

    def optimize(self):
        """
        Do the single scale optimization
        """

        self._set_all_still_missing_parameters()

        # in this way model parameters can be "set" before the optimizer has been properly initialized
        if self.delayed_model_parameters_still_to_be_set:
            print('Setting model parameters, delayed')
            self.set_model_parameters(self.delayed_model_parameters)

        if self.delayed_model_state_dict_still_to_be_set:
            print('Setting model state dict, delayed')
            self.set_model_state_dict(self.delayed_model_state_dict)

        # this allows passing addtional parameters to the smoothers for all models and smoothers
        self.model.set_dictionary_to_pass_to_integrator(self._get_dictionary_to_pass_to_integrator())
        self.criterion.set_dictionary_to_pass_to_smoother(self._get_dictionary_to_pass_to_integrator())

        # optimize for a few steps
        start = time.time()

        self.last_energy = None
        could_not_find_successful_step = False

        if not self._use_external_scheduler:
            self.use_step_size_scheduler = self.params['optimizer'][('use_step_size_scheduler',True,'If set to True the step sizes are reduced if no progress is made')]

            if self.use_step_size_scheduler:
                self.params['optimizer'][('scheduler', {}, 'parameters for the ReduceLROnPlateau scheduler')]
                self.scheduler_verbose = self.params['optimizer']['scheduler'][
                    ('verbose', True, 'if True prints out changes in learning rate')]
                self.scheduler_factor = self.params['optimizer']['scheduler'][('factor', 0.5, 'reduction factor')]
                self.scheduler_patience = self.params['optimizer']['scheduler'][
                    ('patience', 10, 'how many steps without reduction before LR is changed')]

            if self.use_step_size_scheduler and self.scheduler is None:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_instance, 'min',
                                                                            verbose=self.scheduler_verbose,
                                                                            factor=self.scheduler_factor,
                                                                            patience=self.scheduler_patience)

        self.iter_count = 0
        for iter in range(self.nrOfIterations):

            # take a step of the optimizer
            # for p in self.optimizer_instance._params:
            #     p.data = p.data.float()

            current_loss = self.optimizer_instance.step(self._closure)

            # do weight clipping if it is desired
            self._do_weight_clipping()

            # an external scheduler may for example be used in batch optimization
            if not self._use_external_scheduler:
                if self.use_step_size_scheduler:
                    self.scheduler.step(current_loss.data[0])

            if hasattr(self.optimizer_instance,'last_step_size_taken'):
                self.last_successful_step_size_taken = self.optimizer_instance.last_step_size_taken()

            if self.last_successful_step_size_taken==0.0:
                print('Optimizer was not able to find a successful step. Stopping iterations.')
                could_not_find_successful_step = True
                if iter==0:
                    print('The gradient was likely too large or the optimization started from an optimal point.')
                    print('If this behavior is unexpected try adjusting the settings of the similiarity measure or allow the optimizer to try out smaller steps.')

                # to make sure warped images and the map is correct, call closure once more
                self._closure()

            if self.useMap:
                vis_arg = self.rec_phiWarped
            else:
                vis_arg = self.rec_IWarped

            tolerance_reached, was_visualized = self.analysis(self.rec_energy, self.rec_similarityEnergy,
                                                              self.rec_regEnergy, self.rec_opt_par_loss_energy,
                                                              self.rec_recon_energy, self.rec_recon_reg_energy,
                                                              self.reg_rec_joint_loss_energy,
                                                              vis_arg,
                                                              self.rec_custom_optimizer_output_string,
                                                              self.rec_custom_optimizer_output_values)

            if tolerance_reached or could_not_find_successful_step:
                if tolerance_reached:
                    print('Terminating optimization, because the desired tolerance was reached.')

                # force the output of the last image in this case, if it has not been visualized previously
                if not was_visualized and (self.visualize or self.save_fig):
                    _, _ = self.analysis(self.rec_energy, self.rec_similarityEnergy,
                                              self.rec_regEnergy, self.rec_opt_par_loss_energy,
                                              self.rec_recon_energy, self.rec_recon_reg_energy,
                                              self.reg_rec_joint_loss_energy,
                                              vis_arg,
                                              self.rec_custom_optimizer_output_string,
                                              self.rec_custom_optimizer_output_values,
                                              force_visualization=True)
                break

            self.iter_count = iter+1

        if self.show_iteration_output:
            cprint('-->Elapsed time {:.5f}[s]'.format(time.time() - start),  'green')
