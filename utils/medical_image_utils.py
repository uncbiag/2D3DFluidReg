import numpy as np
import pydicom as dicom
import os
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import warnings

from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from skimage import morphology, measure


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

def load_IMG(file_path, shape, spacing, new_spacing):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    mask, bbox = seg_bg_mask(image, True)

    # for i in range(0,10):
    #     plt.imshow((mask*image)[:,:,i*20])
    #     plt.savefig("./log/image_%i.jpg"%i)
    
    # image = win_scale(image, 490, 820, np.float32, [0, 1])
    image = image - 1024
    image[image < -1024] = -1024
    min_intensity = image.min()
    max_intensity = image.max()
    image = (image-image.min())/(max_intensity - min_intensity)

    # image[image>300] = 0
    # image = (image/1000.0+1)*1.673

    return image, mask, bbox

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

def smoother(img, sigma=3):
    D = img.shape[0]
    for i in range(D):
        img[i] = gaussian_filter(img[i], sigma)
    return img

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


def normalize_intensity(img, linear_clip=False):
    if linear_clip:
        img = img - img.min()
        normalized_img =img / np.percentile(img, 95) * 0.95
    else:
        min_intensity = img.min()
        max_intensity = img.max()
        normalized_img = (img-img.min())/(max_intensity - min_intensity)
    # normalized_img = normalized_img*2 - 1
    return normalized_img


def seg_bg_mask(img, only_lung):
    """
    Calculate the segementation mask either for lung only or for the whole body.
    :param img: a 3D image represented in a numpy array.
    :param only_lung: Bool. Indicates whether to segment the lung.
    :return: The segmentation Mask.
    """
    (D,W,H) = img.shape

    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(D/5):int(D/5*4),int(W/5):int(W/5*4),int(H/5):int(H/5*4)] 
    mean = np.mean(middle)  
    img_max = np.max(img)
    img_min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==img_max]=mean
    img[img==img_min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([4,4,4]))
    dilation = morphology.dilation(eroded,np.ones([4,4,4]))

    labels = measure.label(dilation)
    if only_lung:
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        good_labels_bbox = []
        for prop in regions:
            B = prop.bbox
            if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/5 and B[4]<W/20*16 and B[1]>W/10 and
                 B[5]-B[2]<H/20*16 and B[5]-B[2]>H/10 and B[2]>H/10 and B[5]<H/20*18 and
                 B[3]-B[0]>D/4):
                good_labels.append(prop.label)
                good_labels_bbox.append(prop.bbox)
        
        if len(good_labels) == 0:
            good_labels = []
            good_labels_bbox = []
            for prop in regions:
                B = prop.bbox
                if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/6 and B[4]<W/20*18 and B[1]>W/20 and
                    B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20):
                    good_labels.append(prop.label)
                    good_labels_bbox.append(prop.bbox)
        
        if len(good_labels) == 0:
            good_labels = []
            good_labels_bbox = []
            for prop in regions:
                B = prop.bbox
                if B[4]-B[1]<W/20*18 and B[4]-B[1]>W/20 and B[4]<W/20*18 and B[1]>W/20:
                # and B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20:
                    good_labels.append(prop.label)
                    good_labels_bbox.append(prop.bbox)
        
        
        mask = np.ndarray([D,W,H],dtype=np.int8)
        mask[:] = 0

        #
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N, 1, 0)
        
        # Get the bbox of lung
        bbox = [D/2, W/2, H/2, D/2, W/2, H/2]
        for b in good_labels_bbox:
            for i in range(0, 3):
                bbox[i] = min(bbox[i], b[i])
                bbox[i+3] = max(bbox[i+3], b[i+3])
        
        mask = morphology.dilation(mask, np.ones([2,2,2])) # one last dilation
    else:
        mask = np.where(labels==1, 0, 1)
        bbox = [0,0,0,D,W,H]
        # plt.imshow(mask[:,20,:])
        # plt.savefig("./log/temp.jpg")

    return mask, bbox

def binary_dilation(img, radius = 1):
    return morphology.binary_dilation(img, morphology.ball(radius))

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

