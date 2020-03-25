import numpy as np
import pydicom as dicom
import os
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import warnings

from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import normalize


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s, force=True) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePostionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLoacation - slices[1].SliceLoacation)

    for s in slices:
        s.slice_thickness = slice_thickness

    return slices


def resample(imgs, spacing, new_spacing, order=1):
    if len(imgs.shape) == 3 or len(imgs.shape) == 2:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg),[1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def project(img, axis=0):
    # img = torch.ones(img.shape).to(img.device) - img
    return torch.sum(img, axis)


def project_diagonal(img, delta_theta_x):
    device = img.device
    theta_x = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).to(device)*torch.cos(delta_theta_x)+\
              torch.Tensor([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]).to(device)*torch.sin(delta_theta_x)

    aff_grid = F.affine_grid(theta_x[0:3, :].unsqueeze(0), img.shape)
    return torch.sum(F.grid_sample(img, aff_grid, padding_mode="zeros"), 3)


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
  
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
         ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    data_new[data > (wl+ww/2.0)] = out_range[1]-1
    
    return data_new.astype(dtype)


if __name__=="__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dicomPath = '../../Data/Raw/DICOMforMN/S00001/SER00002'
    sdtPath = '../../Data/Raw/NoduleStudyProjections/001/DICOM/'
    processed_file_folder = '../../Data/Preprocessed/sdt0001'
    
    # Processing CT images
    case = load_scan(dicomPath)
    image = np.stack([s.pixel_array for s in case])
    image = image.astype(np.int16)

    for slice_number in range(len(case)):
        intercept = case[slice_number].RescaleIntercept
        slope = case[slice_number].RescaleSlope
            
        # Hounsfield Unit = pixel_value * rescale_slope + rescale_intercept
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)

    case_pixels = np.array(image, dtype=np.float)  # HU value
    spacing = np.array([case[0].SliceThickness, case[0].PixelSpacing[0], case[0].PixelSpacing[1]], dtype=np.float32)  # spacing z,x,y
    case_pixels, new_spacing = resample(case_pixels, spacing, [1, 1, 1])
    case_pixels = np.flip(case_pixels, axis=0).copy() # z is upside down, so flip it

    # case_pixels = win_scale(case_pixels, -650., 100., np.float, [0., 1.])
    # plt.imshow(case_pixels[120], cmap='gray')
    # plt.savefig("./data/case_lung.png")
    # Transform HU to attenuation coefficient u
    case_pixels = (case_pixels/1000.0+1)*1.673
    
    # Save preprocessed CT to numpy
    if not os.path.exists(processed_file_folder):
        os.mkdir(processed_file_folder)
    np.save(processed_file_folder+"/ct.npy", case_pixels)
    
    # Processing raw data from sDT
    image = [dicom.read_file(sdtPath + '/' + s).pixel_array for s in os.listdir(sdtPath)]
    image = np.array(image)
    image = image.astype(np.float32)
    image = image[:, 0:2052]
    image = -np.log(image/65535+0.0001)

    
    # proj_y[proj_y==0]=1
    # proj_y[proj_y>16000] = 16000
    # proj_y = -proj_y
    # # proj_y = np.log(proj_y)
    # # proj_y[proj_y<7] = 7
    # # proj_y = -proj_y
    # proj_y_max = np.max(proj_y, axis=(1,2))
    # proj_y_min = np.min(proj_y, axis=(1,2))
    # dur = proj_y_max - proj_y_min
    # for i in range(proj_y.shape[0]):
    #     proj_y[i] = (proj_y[i] - proj_y_min[i])/dur[i]*255
    np.save(processed_file_folder+"/projection.npy", image)

    # Compare projection of CT and sDT
    # case_torch = torch.from_numpy(case_pixels).to(device).float()
    # tan_theta_x = np.arctan(138.16/1345.1434)
    # theta_x = torch.tensor([tan_theta_x]).to(device)
    # proj_x = project_diagonal(case_torch.unsqueeze(0).unsqueeze(0), theta_x)[0, 0].cpu().numpy()
    # proj_x_max = np.max(proj_x)
    # proj_x_min = np.min(proj_x)
    # proj_x = (proj_x - proj_x_min)/(proj_x_max - proj_x_min)*255
    # I0_np = proj_x.astype(np.uint8)

    # sample_processed = F.interpolate(torch.from_numpy(proj_y[0]).unsqueeze(0).unsqueeze(0), proj_x.shape)
    # sample_processed_np = sample_processed.data.numpy()[0, 0]
    # I1_np = sample_processed_np.astype(np.uint8)

    # # Compute the joint level of surprises
    # h_I0I1, xedges, yedges = np.histogram2d(I0_np.ravel(), I1_np.ravel(), bins=256)
    # h_I0I1 = h_I0I1/(I0_np.shape[0]*I0_np.shape[1])
    # surprises = -np.log2(h_I0I1, where=np.where(h_I0I1!=0, True, False))

    # figs, axes = plt.subplots(3, 2)
    # axes[0,0].hist(I0_np)
    # axes[0,1].imshow(I0_np, cmap='gray')
    # axes[1,0].hist(I1_np)
    # axes[1,1].imshow(I1_np, cmap='gray')
    # axes[2,0].imshow(I0_np-I1_np)
    # axes[2,1].imshow(surprises, cmap='gray')
    # plt.savefig("./data/ctplayground.png")
