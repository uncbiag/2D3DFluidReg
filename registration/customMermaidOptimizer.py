from mermaid.multiscale_optimizer import SimpleRegistration, MultiScaleRegistrationOptimizer, SingleScaleRegistrationOptimizer
import numpy as np
import os
import mermaid.utils as utils
from termcolor import cprint
import mermaid.visualize_registration_results as vizReg

class SimpleMultiScaleRegistrationFor2D3D(SimpleRegistration):
    """
    Simple multi scale registration
    """
    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(SimpleMultiScaleRegistrationFor2D3D, self).__init__(ISource, ITarget, spacing,sz,params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.optimizer = MultiScaleRegistrationOptimizerFor2D3D(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource,self.ITarget)

class SimpleSingleScaleRegistrationFor2D3D(SimpleRegistration):
    """
    Simple multi scale registration
    """
    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(SimpleSingleScaleRegistrationFor2D3D, self).__init__(ISource, ITarget, spacing,sz,params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.optimizer = SingleScaleRegistrationOptimizerFor2D3D(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource,self.ITarget)

class SingleScaleRegistrationOptimizerFor2D3D(SingleScaleRegistrationOptimizer):
    """
    Custumize the single scale registration optimizer for 2D/3D. Major differences are in analysis method.
    """
    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None):
        super(SingleScaleRegistrationOptimizerFor2D3D, self).__init__(sz, spacing, useMap, mapLowResFactor, params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def compute_low_res_image_if_needed(self):
        """To be called before the optimization starts"""
        if self.multi_scale_info_dic is None:
            ISource = self.ISource
            ITarget = self.ITarget
            LSource = self.LSource
            LTarget = self.LTarget
            spacing = self.spacing
        else:
            ISource, ITarget, LSource, LTarget, spacing = self.multi_scale_info_dic['ISource'], self.multi_scale_info_dic['ITarget'],\
                                                          self.multi_scale_info_dic['LSource'],self.multi_scale_info_dic['LTarget'],self.multi_scale_info_dic['spacing']
        if self.mapLowResFactor is not None:
            self.lowResISource = self._compute_low_res_image(ISource,self.params,spacing)
            # todo: can be removed to save memory; is more experimental at this point
            # TODO: Lin- comment out for the purpose of 3d 2d registration
            # self.lowResITarget = self._compute_low_res_image(ITarget,self.params,spacing)
            if self.LSource is not None and self.LTarget is not None:
                self.lowResLSource = self._compute_low_res_label_map(LSource,self.params,spacing)
                self.lowResLTarget = self._compute_low_res_label_map(LTarget, self.params,spacing)

    def analysis(self, energy, similarityEnergy, regEnergy, opt_par_energy, phi_or_warped_image, custom_optimizer_output_string ='', custom_optimizer_output_values=None, force_visualization=False):
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

        if custom_optimizer_output_values is not None:
            for key in custom_optimizer_output_values:
                self._add_to_history(key,custom_optimizer_output_values[key])

        if self.last_energy is not None:

            # relative function tolerance: |f(xi)-f(xi+1)|/(1+|f(xi)|)
            self.rel_f = abs(self.last_energy - cur_energy) / (1 + abs(cur_energy))
            self._add_to_history('relF', self.rel_f[0])

            if self.show_iteration_output:
                cprint('{iter:5d}-Tot: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | optParE={optParE:08.4f} | relF={relF:08.4f} | {cos}'
                       .format(iter=self.iter_count,
                               energy=utils.get_scalar(cur_energy),
                               similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())),
                               regE=utils.get_scalar(utils.t2np(regEnergy.float())),
                               optParE=utils.get_scalar(utils.t2np(opt_par_energy.float())),
                               relF=utils.get_scalar(self.rel_f),
                               cos=custom_optimizer_output_string), 'red')
                cprint('{iter:5d}-Img: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} |'
                       .format(iter=self.iter_count,
                               energy=utils.get_scalar(cur_energy) / current_batch_size,
                               similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())) / current_batch_size,
                               regE=utils.get_scalar(utils.t2np(regEnergy.float())) / current_batch_size), 'blue')

            # check if relative convergence tolerance is reached
            if self.rel_f < self.rel_ftol:
                if self.show_iteration_output:
                    print('Reached relative function tolerance of = ' + str(self.rel_ftol))
                reached_tolerance = True

        else:
            self._add_to_history('relF', None)
            if self.show_iteration_output:
                cprint('{iter:5d}-Tot: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | optParE={optParE:08.4f} | relF=  n/a    | {cos}'
                      .format(iter=self.iter_count,
                              energy=utils.get_scalar(cur_energy),
                              similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())),
                              regE=utils.get_scalar(utils.t2np(regEnergy.float())),
                              optParE=utils.get_scalar(utils.t2np(opt_par_energy.float())),
                              cos=custom_optimizer_output_string), 'red')
                cprint('{iter:5d}-Img: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} |'
                      .format(iter=self.iter_count,
                              energy=utils.get_scalar(cur_energy)/current_batch_size,
                              similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float()))/current_batch_size,
                              regE=utils.get_scalar(utils.t2np(regEnergy.float()))/current_batch_size),'blue')

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
            if self.save_fig:
                visual_param['save_fig_path'] = self.save_fig_path
                visual_param['save_fig_path_byname'] = os.path.join(self.save_fig_path, 'byname')
                visual_param['save_fig_path_byiter'] = os.path.join(self.save_fig_path, 'byiter')
                visual_param['save_fig_num'] = self.save_fig_num
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
                        
                        # Plot for 2D/3D registration
                        if self.params['model']['registration_model']['similarity_measure']['type'] == 'projection':
                            pose = np.array(self.params['model']['registration_model']['emitter_pos_scale_list'][0:1])
                            sample_rate = self.params['model']['registration_model']['sample_rate']
                            resolution = self.ITarget.shape[2:]
                            i0 = self.criterion.similarityMeasure.project(self.ISource, pose, resolution, sample_rate)
                            iwarped = self.criterion.similarityMeasure.project(I1Warped, pose, resolution, sample_rate)
                            itarget = self.ITarget[:, 0, :, :].unsqueeze(0)
                            visual_param["prefix"] = self.params['model']['registration_model']['type']
                        else:
                            i0 = self.ISource
                            itarget = self.ITarget
                            iwarped = I1Warped

                        vizReg.show_current_images(iter=iter_count,
                                                iS=i0,
                                                iT=itarget,
                                                iW=iwarped,
                                                iSL=None,
                                                iTL=None,
                                                iWL=None,
                                                vizImages=vizImage,
                                                vizName=vizName,
                                                #phiWarped=phi_or_warped_image,
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

class MultiScaleRegistrationOptimizerFor2D3D(MultiScaleRegistrationOptimizer):
    """
    Class to perform multi-scale optimization. Essentially puts a loop around multiple calls of the
    single scale optimizer and starts with the registration of downsampled images. When moving up
    the hierarchy, the registration parameters are upsampled from the solution at the previous lower resolution
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None ):
        super(MultiScaleRegistrationOptimizerFor2D3D, self).__init__(sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
    
    def optimize(self):
        """
        Perform the actual multi-scale optimization
        """
        self._set_all_still_missing_parameters()

        if (self.ISource is None) or (self.ITarget is None):
            raise ValueError('Source and target images need to be set first')

        upsampledParameters = None
        upsampledParameterSpacing = None
        upsampledSz = None
        lastSuccessfulStepSizeTaken = None

        nrOfScales = len(self.scaleFactors)

        # check that we have the right number of iteration parameters
        assert (nrOfScales == len(self.scaleIterations))

        print('Performing multiscale optmization with scales: ' + str(self.scaleFactors))

        # go from lowest to highest scale
        reverseScales = self.scaleFactors[-1::-1]
        reverseIterations = self.scaleIterations[-1::-1]
        over_scale_iter_count = 0

        for en_scale in enumerate(reverseScales):
            print('Optimizing for scale = ' + str(en_scale[1]))

            # create the images
            currentScaleFactor = en_scale[1]
            currentScaleNumber = en_scale[0]

            currentDesiredSz = self._get_desired_size_from_scale(self.ISource.size(), currentScaleFactor)
            currentDesiredSzProj = self._get_desired_size_from_scale(self.ITarget.size(), currentScaleFactor)

            currentNrOfIteratons = reverseIterations[currentScaleNumber]

            ISourceC, spacingC = self.sampler.downsample_image_to_size(self.ISource, self.spacing, currentDesiredSz[2::],self.spline_order)

            # TODO: A quick way to test sDCT CT registration, should write its own multiscale registration class later
            # ISourceC, spacingC = self.sampler.downsample_image_by_factor(self.ISource, self.spacing, currentScaleFactor)
            # ISourceC = ISourceC.to(torch.device("cuda"))
            ITargetC, spacingT = self.sampler.downsample_image_to_size(self.ITarget, self.spacing[0::2], currentDesiredSzProj[2::],self.spline_order)
            #ITargetC, spacingC = self.sampler.downsample_image_to_size(self.ITarget, self.spacing, currentDesiredSz[2::],self.spline_order)
            LSourceC = None
            LTargetC = None
            if self.LSource is not None and self.LTarget is not None:
                LSourceC, spacingC = self.sampler.downsample_image_to_size(self.LSource, self.spacing, currentDesiredSz[2::],0)
                LTargetC, spacingC = self.sampler.downsample_image_to_size(self.LTarget, self.spacing, currentDesiredSzProj[2::],0)
            initialMap = None
            initialInverseMap = None
            weight_map=None
            if self.initialMap is not None:
                initialMap,_ = self.sampler.downsample_image_to_size(self.initialMap,self.spacing, currentDesiredSz[2::],self.spline_order,zero_boundary=False)
            if self.initialInverseMap is not None:
                initialInverseMap,_ = self.sampler.downsample_image_to_size(self.initialInverseMap,self.spacing, currentDesiredSz[2::],self.spline_order,zero_boundary=False)
            if self.weight_map is not None:
                weight_map,_ =self.sampler.downsample_image_to_size(self.weight_map,self.spacing, currentDesiredSz[2::],self.spline_order,zero_boundary=False)
            szC = np.array(ISourceC.size())  # this assumes the BxCxXxYxZ format
            mapLowResFactor = None if currentScaleNumber==0 else self.mapLowResFactor
            self.ssOpt = SingleScaleRegistrationOptimizerFor2D3D(szC, spacingC, self.useMap, mapLowResFactor, self.params, compute_inverse_map=self.compute_inverse_map,default_learning_rate=self.default_learning_rate)
            print('Setting learning rate to ' + str( lastSuccessfulStepSizeTaken ))
            self.ssOpt.set_last_successful_step_size_taken( lastSuccessfulStepSizeTaken )
            self.ssOpt.set_initial_map(initialMap,initialInverseMap)

            if ((self.add_model_name is not None) and
                    (self.add_model_networkClass is not None) and
                    (self.add_model_lossClass is not None)):
                self.ssOpt.add_model(self.add_model_name, self.add_model_networkClass, self.add_model_lossClass, use_map=self.add_model_use_map)

            # now set the actual model we want to solve
            self.ssOpt.set_model(self.model_name)
            if weight_map is not None:
                self.ssOpt.set_initial_weight_map(weight_map,self.freeze_weight)


            if (self.addSimName is not None) and (self.addSimMeasure is not None):
                self.ssOpt.add_similarity_measure(self.addSimName, self.addSimMeasure)

            # setting the optimizer
            if self.optimizer is not None:
                self.ssOpt.set_optimizer(self.optimizer)
                self.ssOpt.set_optimizer_params(self.optimizer_params)
            elif self.optimizer_name is not None:
                self.ssOpt.set_optimizer_by_name(self.optimizer_name)

            self.ssOpt.set_rel_ftol(self.get_rel_ftol())

            self.ssOpt.set_visualization(self.get_visualization())
            self.ssOpt.set_visualize_step(self.get_visualize_step())
            self.ssOpt.set_n_scale(en_scale[1])
            self.ssOpt.set_over_scale_iter_count(over_scale_iter_count)

            if self.get_save_fig():
                self.ssOpt.set_expr_name(self.get_expr_name())
                self.ssOpt.set_save_fig(self.get_save_fig())
                self.ssOpt.set_save_fig_path(self.get_save_fig_path())
                self.ssOpt.set_save_fig_num(self.get_save_fig_num())
                self.ssOpt.set_pair_name(self.get_pair_name())
                self.ssOpt.set_n_scale(en_scale[1])
                self.ssOpt.set_source_label(self.get_source_label())
                self.ssOpt.set_target_label(self.get_target_label())


            self.ssOpt.set_source_image(ISourceC)
            self.ssOpt.set_target_image(ITargetC)
            self.ssOpt.set_multi_scale_info(self.ISource,self.ITarget,self.spacing,self.LSource,self.LTarget)
            if self.LSource is not None and self.LTarget is not None:
                self.ssOpt.set_source_label(LSourceC)
                self.ssOpt.set_target_label(LTargetC)

            if upsampledParameters is not None:
                # check that the upsampled parameters are consistent with the downsampled images
                spacingError = False
                expectedSpacing = None

                if mapLowResFactor is not None:
                    expectedSpacing = utils._get_low_res_spacing_from_spacing(spacingC, szC, upsampledSz)
                    # the spacing of the upsampled parameters will be different
                    if not (abs(expectedSpacing - upsampledParameterSpacing) < 0.000001).all():
                        spacingError = True
                elif not (abs(spacingC - upsampledParameterSpacing) < 0.000001).all():
                    expectedSpacing = spacingC
                    spacingError = True

                if spacingError:
                    print(expectedSpacing)
                    print(upsampledParameterSpacing)
                    raise ValueError('Upsampled parameters and downsampled images are of inconsistent dimension')

                # now that everything is fine, we can use the upsampled parameters
                print('Explicitly setting the optimization parameters')
                self.ssOpt.set_model_parameters(upsampledParameters)

            # do the actual optimization
            print('Optimizing for at most ' + str(currentNrOfIteratons) + ' iterations')
            self.ssOpt._set_number_of_iterations_from_multi_scale(currentNrOfIteratons)
            self.ssOpt.optimize()

            self._add_to_history('scale_nr',currentScaleNumber)
            self._add_to_history('scale_factor',currentScaleFactor)
            self._add_to_history('ss_history',self.ssOpt.get_history())

            lastSuccessfulStepSizeTaken = self.ssOpt.get_last_successful_step_size_taken()
            over_scale_iter_count += currentNrOfIteratons

            # if we are not at the very last scale, then upsample the parameters
            if currentScaleNumber != nrOfScales - 1:
                # we need to revert the downsampling to the next higher level
                scaleTo = reverseScales[currentScaleNumber + 1]
                upsampledSz = self._get_desired_size_from_scale(self.ISource.size(), scaleTo)
                print('Before')
                print(upsampledSz)
                if self.useMap:
                    if self.mapLowResFactor is not None:
                        # parameters are upsampled differently here, because they are computed at low res
                        upsampledSz = utils._get_low_res_size_from_size(upsampledSz,self.mapLowResFactor)
                        print(self.mapLowResFactor)
                        print('After')
                        print(upsampledSz)
                upsampledParameters, upsampledParameterSpacing = self.ssOpt.upsample_model_parameters(upsampledSz[2::])