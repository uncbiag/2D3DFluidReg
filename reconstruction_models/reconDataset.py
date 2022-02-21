
import os

import torch
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
from multiprocessing import *
import blosc
import progressbar as pb
import SimpleITK as sitk


class dirlabDataset(Dataset):
    def __init__(self, data_path, data_id_path, max_num_for_loading=10, device='cpu'):
        super(dirlabDataset, self).__init__()
        self.data_path = data_path + "/preprocessed"
        self.data_id_path = data_id_path
        self.volume_list = []
        self.proj_list = []
        self.device = device

        self.max_num_for_loading = max_num_for_loading
        self._init_id_list()
        self._load_data_pool()
    
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(blosc.unpack_array(self.volume_list[idx])), 
                torch.from_numpy(blosc.unpack_array(self.proj_list[idx])),
                idx)

    
    def _init_id_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        if not os.path.exists(self.data_id_path):
            self.id_list = []
            return
        self.id_list = np.load(self.data_id_path)

        if self.max_num_for_loading > 0:
            read_num = min(self.max_num_for_loading, len(self.id_list))
            self.id_list = self.id_list[:read_num]

        if len(self.id_list) == 0:
            self.id_list = ['pair_{}'.format(idx) for idx in range(len(self.id_list))]

    def _load_data_pool(self):
        manager = Manager()
        data_dict = manager.dict()
        if self.max_num_for_loading == 1:
            num_of_workers = 1
        else:
            num_of_workers = 12
            num_of_workers = num_of_workers if len(self.id_list)>12 else 2
        split_id_list = self.__split_dict(self.id_list, num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(target=self._load_data, args=(split_id_list[i], data_dict,))
            p.start()
            print("pid:{} start:".format(p.pid))

            procs.append(p)

        for p in procs:
            p.join()

        print("the loading phase finished, total {} img and labels have been loaded".format(len(data_dict)))

        for id in self.id_list:
            data = data_dict[id]
            self.volume_list.append(data['3d'])
            self.proj_list.append(data['proj'])

    def _load_data(self, case_id_list, data_list):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(case_id_list)).start()
        count = 0
        for case_id in case_id_list:
            data = {}
            img = np.load(os.path.join(self.data_path, case_id+"_source.npy")).astype(np.float32)
            img = self._calc_relative_atten_coef(img)
            # img = self._normalize_intensity(img)
            data['3d'] = blosc.pack_array(img)
            
            proj = np.load(os.path.join(self.data_path, case_id+"_source_proj.npy")).astype(np.float32)
            # proj = self._normalize_intensity(proj)
            data['proj'] = blosc.pack_array(proj)
            
            data_list[case_id] = data
            count += 1
            pbar.update(count)
        pbar.finish()

    def __split_dict(self, dict_to_split, split_num):
        index_list = list(range(len(dict_to_split)))
        index_split = np.array_split(np.array(index_list), split_num)
        split_dict = []
        for i in range(split_num):
            dj = dict_to_split[index_split[i][0]:index_split[i][-1]+1]
            split_dict.append(dj)
        return split_dict

    def _resize_img(self, img, new_size, is_label=False):
        """
        :param img: sitk input, factor is the outputs_ize/patched_sized
        :return:
        """
        dimension = len(img.shape)
        img = sitk.GetImageFromArray(img)
        img_sz = img.GetSize()
        
        resize_factor = np.array(new_size)/np.flipud(img_sz)
        resize = not all([factor == 1 for factor in resize_factor])
        if resize:
            resampler= sitk.ResampleImageFilter()
            factor = np.flipud(resize_factor)
            affine = sitk.AffineTransform(dimension)
            matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
            after_size = [round(img_sz[i]*factor[i]) for i in range(dimension)]
            after_size = [int(sz) for sz in after_size]
            matrix[0, 0] =1./ factor[0]
            matrix[1, 1] =1./ factor[1]
            if dimension == 3:
                matrix[2, 2] =1./ factor[2]
            affine.SetMatrix(matrix.ravel())
            resampler.SetSize(after_size)
            resampler.SetTransform(affine)
            if is_label:
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            else:
                resampler.SetInterpolator(sitk.sitkBSpline)
            img_resampled = resampler.Execute(img)
        else:
            img_resampled = img
        return sitk.GetArrayFromImage(img_resampled)

    def _normalize_intensity(self, img, linear_clip=False):
        # TODO: Lin-this line is for CT. Modify it to make this method more general.
        if linear_clip:
            img = img - img.min()
            normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        # normalized_img = normalized_img*2 - 1
        return normalized_img

    def _calc_relative_atten_coef(self, img):
        new_img = img.astype(np.float32).copy()
        new_img[new_img<-1024] = -1024
        return (new_img+1024.)/1024.