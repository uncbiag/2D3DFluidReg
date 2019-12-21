
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from test_projection import project_grid
from CTPlayground import resample

import matplotlib.pyplot as plt

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
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([1,1,1]))
    dilation = morphology.dilation(eroded,np.ones([2,2,2]))

    labels = measure.label(dilation)
    if only_lung:
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[4]-B[1]<W/20*19 and B[5]-B[2]<H/20*19 and B[2]>W/20 and B[5]<W/20*19:
                good_labels.append(prop.label)
        mask = np.ndarray([D,W,H],dtype=np.int8)
        mask[:] = 0

        #
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([2,2,2])) # one last dilation
    else:
        mask = np.where(labels==1, 0, 1)
    return mask


def load_IMG(file_path, shape, spacing, new_spacing):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    mask = seg_bg_mask(image, False)
    image = mask*image

    image, spacing = resample(image, np.array(spacing), new_spacing)
    image = image.astype(np.float32)

    # image[image>300] = 0
    # image = (-image/1000.0+1)*1.673

    return image


def calculate_projection(img, poses, resolution_scale, sample_rate, device):
    poses = poses*img.shape
    spacing = [1., 1., 1.]
    I1 = torch.from_numpy(img).to(device).float()
    I1 = I1.unsqueeze(0).unsqueeze(0)
    resolution = [int(I1.shape[2] * resolution_scale),
                  int(I1.shape[3] * resolution_scale)]
    projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
    for i in range(poses.shape[0]):
        grid = torch.flip(project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing),[3])
        projections[0, i] = (torch.sum(F.grid_sample(I1, grid.unsqueeze(0), align_corners=True), dim=4)[0, 0])
        del grid
        torch.cuda.empty_cache()

    return projections[0].detach().cpu().numpy()

def smoother(img, sigma=3):
    D = img.shape[0]
    for i in range(D):
        img[i] = gaussian_filter(img[i], sigma)
    return img

def preprocessData(source_file, target_file, dest_folder, shape, spacing,
                   new_spacing, calc_projection=False, poses=[],
                   resolution_scale=1.0, sample_rate=[1, 1], show_projection=False):
    print("Preprocessing data...")

    img_0 = load_IMG(source_file, shape, spacing, new_spacing)
    np.save(dest_folder + "/I0_3d.npy", img_0)

    img_1 = load_IMG(target_file, shape, spacing, new_spacing)
    np.save(dest_folder + "/I1_3d.npy", img_1)

    if calc_projection:
        device = torch.device("cuda")
        img_proj_0 = calculate_projection(img_0, poses, resolution_scale,
                                          sample_rate, device)
        np.save(dest_folder + "/I0_proj.npy", img_proj_0)

        img_proj_1 = calculate_projection(img_1, poses, resolution_scale,
                                          sample_rate, device)
        np.save(dest_folder + "/I1_proj.npy", img_proj_1)

        if show_projection:
            step = min(1, int(img_proj_0.shape[0]/5))
            fig, ax = plt.subplots(2, step*5)
            for i in range(0, 5):
                ax[0, i].imshow(img_proj_0[i*step])
                ax[1, i].imshow(img_proj_1[i*step])
            # plt.savefig("./data/projections.png")
            plt.show()

if __name__ == "__main__":
    pass
