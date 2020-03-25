
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

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
                print(B)
        
        if len(good_labels) == 0:
            good_labels = []
            good_labels_bbox = []
            for prop in regions:
                B = prop.bbox
                if (B[4]-B[1]<W/20*18 and B[4]-B[1]>W/6 and B[4]<W/20*18 and B[1]>W/20 and
                    B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20):
                    good_labels.append(prop.label)
                    good_labels_bbox.append(prop.bbox)
                    print("2")
                    print(B)
        
        if len(good_labels) == 0:
            good_labels = []
            good_labels_bbox = []
            for prop in regions:
                B = prop.bbox
                if B[4]-B[1]<W/20*18 and B[4]-B[1]>W/20 and B[4]<W/20*18 and B[1]>W/20:
                # and B[5]-B[2]<H/20*18 and B[5]-B[2]>H/20:
                    good_labels.append(prop.label)
                    good_labels_bbox.append(prop.bbox)
                    print("3")
                    print(B)
        
        
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

def showAll(img):
    amount = img.shape[0]
    rowCount = int(amount/10)
    if amount%10 != 0:
        rowCount = rowCount + 1
    fig, axes = plt.subplots(rowCount, 10)
    for i in range(0, rowCount):
        for j in range(0, 10):
            index = i * 10 + j
            if index >= amount: 
                break
            axes[i, j].imshow(img[index])
    plt.show()


def load_IMG(file_path, shape, spacing, new_spacing):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    mask, bbox = seg_bg_mask(image, True)

    # for i in range(0,10):
    #     plt.imshow((mask*image)[:,:,i*20])
    #     plt.savefig("./log/image_%i.jpg"%i)
    
    image = image.astype(np.float32)
    image_max = np.max(image)
    image_min = np.min(image)
    image = image/(image_max - image_min)

    # image[image>300] = 0
    # image = (image/1000.0+1)*1.673

    return image, mask, bbox


def project_grid(img, emi_pos, resolution, sample_rate, obj_shape, spacing):
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
    dx = torch.norm(dx*spacing.unsqueeze(0).unsqueeze(0), dim=2)

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

def calculate_projection(img, poses_scale, resolution_scale, sample_rate, spacing, device):
    poses = poses_scale*img.shape[1]
    spacing = torch.tensor(spacing).to(device)
    I1 = torch.from_numpy(img).to(device)
    I1 = I1.unsqueeze(0).unsqueeze(0)
    resolution = [int(I1.shape[2] * resolution_scale),
                  int(I1.shape[4] * resolution_scale)]
    projections = torch.zeros((1, poses.shape[0], resolution[0]*sample_rate[0], resolution[1]*sample_rate[2])).to(device)
    for i in range(poses.shape[0]):
        grid, dx = project_grid(I1, poses[i], (resolution[0], resolution[1]), sample_rate, I1.shape[2:], spacing)
        grid = torch.flip(grid,[3])
        dx = dx.unsqueeze(0).unsqueeze(0)
        projections[0, i] = torch.mul(torch.sum(F.grid_sample(I1, grid.unsqueeze(0), align_corners=True), dim=4), dx)[0, 0]
        # np.save("./log/grids_sim_matrix_"+str(i)+".npy", grid.cpu().numpy())
        del grid
        torch.cuda.empty_cache()
        
    return projections[0].detach().cpu().numpy()

def smoother(img, sigma=3):
    D = img.shape[0]
    for i in range(D):
        img[i] = gaussian_filter(img[i], sigma)
    return img

def preprocessData(source_file, target_file, dest_folder, dest_prefix, shape, spacing,
                   new_spacing, smooth=False, sigma=6, calc_projection=False, poses_scale=[],
                   resolution_scale=1.0, sample_rate=[1, 1], show_projection=False):
    print("Preprocessing data...")

    img_0, mask_0, bbox_0 = load_IMG(source_file, shape, spacing, new_spacing)
    img_1, mask_1, bbox_1 = load_IMG(target_file, shape, spacing, new_spacing)

    # Figure out the bbox size
    bbox = np.ndarray((6)).astype(np.int)
    for i in range(0,3):
        bbox[i] = min(bbox_0[i], bbox_1[i])
        bbox[i+3] = max(bbox_0[i+3], bbox_1[i+3])
    prop = {'crop': bbox}
    prop["dim"] = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]

    # Crop and resample the image
    img_0 = img_0 * mask_0
    img_0 = img_0.astype(np.float32)
    img_0 = img_0[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    img_0, _ = resample(img_0, np.array(spacing), new_spacing)

    img_1 = img_1 * mask_1
    img_1 = img_1.astype(np.float32)
    img_1 = img_1[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    img_1, _ = resample(img_1, np.array(spacing), new_spacing)

    # Smooth the image
    if smooth:
        img_0 = smoother(img_0, sigma=sigma)
        img_1 = smoother(img_1, sigma=sigma)

    # Save the 3d image
    np.save(dest_folder + "/" + dest_prefix + "_I0_3d.npy", img_0)
    np.save(dest_folder + "/" + dest_prefix + "_I1_3d.npy", img_1)
    
    np.save(dest_folder + "/" + dest_prefix + "_prop.npy", prop)

    # Calculate the projection image
    if calc_projection:
        device = torch.device("cuda")
        img_proj_0 = calculate_projection(img_0, poses_scale, resolution_scale,
                                          sample_rate, new_spacing, device)
        np.save(dest_folder + "/" + dest_prefix + "_I0_proj.npy", img_proj_0)

        img_proj_1 = calculate_projection(img_1, poses_scale, resolution_scale,
                                          sample_rate, new_spacing, device)
        np.save(dest_folder + "/" + dest_prefix + "_I1_proj.npy", img_proj_1)

        if show_projection:
            step = min(1, int(img_proj_0.shape[0]/10))
            fig, ax = plt.subplots(2, img_proj_0.shape[0])
            for i in range(0, img_proj_0.shape[0]):
                ax[0, i].imshow(img_proj_0[i])
                ax[1, i].imshow(img_proj_1[i])
            ax[0,0].set_ylabel("Source projection")
            ax[1,0].set_ylabel("Target projection")
            plt.savefig("./log/" + dest_prefix + "_projections.png", dpi=300)
            # plt.show()

if __name__ == "__main__":
    pass
