from mermaid import viewers
import mermaid
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import os
import torch
from mermaid import image_sampling
from CTPlayground import resample

def plot_grid(ax, gridx,gridy, **kwargs):
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:,i], gridy[:,i], **kwargs)

class ImageViewer3D_Sliced_Grids(viewers.ImageViewer3D_Sliced):
    """
    Specialization of 3D sliced viewer to also display contours
    """
    def __init__(self, ax, data, phi, sliceDim, textStr='Slice', showColorbar=False):
        """
        Constructor
        :param ax: axis
        :param data: data (image array, XxYxZ)
        :param phi: map (dimxXxYxZ)
        :param sliceDim: slice dimension
        :param textStr: title string
        :param showColorbar: (bool) show colorbar
        """
        self.phi = phi
        self.d, self.w, self.h = data.shape
        """map"""
        super(ImageViewer3D_Sliced_Grids,self).__init__(ax,data, sliceDim, textStr, showColorbar)

    def get_phi_slice_at_dimension(self,index):
        """
        Get map (based on which we can draw contours) at a particular slice index
        
        :param index: slice index 
        :return: returns the map at this slice index
        """
        # slicing a along a given dimension at index, index
        slc = [slice(None)] * len(self.phi.shape)
        slc[self.sliceDim+1] = slice(index, index+1)
        
        return (self.phi[tuple(slc)]).squeeze()

    def show_grids(self):
        """
        display the contours for a particular slice
        """
        plt.sca(self.ax)
        phiSliced = self.get_phi_slice_at_dimension(self.index)
        for d in range(0,self.sliceDim):
            plt.contour(phiSliced[d,:,:], np.linspace(0,self.data.shape[d],20),colors='red',linestyles='solid', linewidths=0.7)
        for d in range(self.sliceDim+1,3):
            plt.contour(phiSliced[d,:,:], np.linspace(0,self.data.shape[d],20),colors='red',linestyles='solid', linewidths=0.7)

    def previous_slice(self):
        """
        display previous slice
        """
        super(ImageViewer3D_Sliced_Grids,self).previous_slice()
        self.show_grids()

    def next_slice(self):
        """
        display next slice
        """
        super(ImageViewer3D_Sliced_Grids,self).next_slice()
        self.show_grids()

    def set_synchronize(self, index):
        """
        set slice to a particular index (to synchronize views)

        :param index: slice index 
        """
        super(ImageViewer3D_Sliced_Grids,self).set_synchronize(index)
        self.show_grids()

    def show(self):
        """
        Show the image with contours overlaid
        """
        super(ImageViewer3D_Sliced_Grids,self).show()
        self.show_grids()


I0_file = "./data/input_svf.npy"
disp_file = "./data/disp_svf_double.npy"
warped_file = "./data/warped_svf_double.npy"
sDT_dicom = "../../Data/Raw/DICOMforMN/DICOM/S00002/SER00001"
ct_IMG = "../eval_data/copd1/copd1/copd1_eBHCT.img"
demo_npy = "../eval_data/preprocessed/ehale_3d.npy"

SHOW_SDT = False
SHOW_CT_IMG = False
SHOW_DISP = True
SHOW_DEMO_NPY = True

spacing = np.array([72, 80, 80])
I0 = np.load(I0_file)[0,0]
phi = np.swapaxes(np.multiply(np.swapaxes(np.load(disp_file)[0],0,3), spacing), 0, 3)
warped = np.load(warped_file)[0,0]
d, w, h = I0.shape

if SHOW_SDT:
    file_list = os.listdir(sDT_dicom)
    file_list.sort()
    case  = [dicom.read_file(sDT_dicom + '/' + s) for s in file_list]
    image = np.stack([s.pixel_array for s in case])
    image = np.transpose(image, (1,0,2))
    image = image[:2100,:,:]
    image = image.astype(np.float32)
    I1 = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(torch.device("cuda"))
    spacing = np.array([1., 1., 1.])
    sampler = image_sampling.ResampleImage()
    sdt, spacing = sampler.downsample_image_by_factor(I1, spacing, scalingFactor=0.48)
    sdt = sdt.detach().cpu().numpy()[0,0]

if SHOW_CT_IMG:
    shape = (121, 512, 512)
    spacing = [2.5, 0.625, 0.625]
    dtype = np.dtype("<i2")
    fid = open(ct_IMG, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)
    ct_img, spacing = resample(image, np.array(spacing), [1,1,1])
    image = image.astype(np.float32)
    ct_img[ct_img<0] = 1000
    ct_img = (-ct_img/1000.0+1)*1.673
    I1 = torch.from_numpy(ct_img).unsqueeze(0).unsqueeze(0).to(torch.device("cuda"))
    spacing = np.array([1., 1., 1.])
    sampler = image_sampling.ResampleImage()
    ct_img, spacing = sampler.downsample_image_by_factor(I1, spacing, scalingFactor=0.2)
    ct_img = ct_img.detach().cpu().numpy()[0,0]

if SHOW_DEMO_NPY:
    image = np.load(demo_npy)

fig,ax = plt.subplots(3, 3)

plt.setp(plt.gcf(), 'facecolor', 'white')
plt.style.use('bmh')


ivx = viewers.ImageViewer3D_Sliced(ax[0,0], I0, 0, 'Source Image - Z slice')
ivy = viewers.ImageViewer3D_Sliced(ax[0,1], I0, 1, 'Source Image - X slice')
ivz = viewers.ImageViewer3D_Sliced(ax[0,2], I0, 2, 'Source Image - Y slice')

if not SHOW_DISP:
    warped_ivx = viewers.ImageViewer3D_Sliced(ax[1,0], warped, 0, 'Warped Image - Z slice')
    warped_ivy = viewers.ImageViewer3D_Sliced(ax[1,1], warped, 1, 'arped Image - X slice')
    warped_ivz = viewers.ImageViewer3D_Sliced(ax[1,2], warped, 2, 'arped Image - Y slice')
else:
    warped_ivx = ImageViewer3D_Sliced_Grids(ax[1,0], warped, phi, 0, 'arped Image - Z slice')
    warped_ivy = ImageViewer3D_Sliced_Grids(ax[1,1], warped, phi, 1, 'arped Image - X slice')
    warped_ivz = ImageViewer3D_Sliced_Grids(ax[1,2], warped, phi, 2, 'arped Image - Y slice')

if SHOW_SDT:
    warped_ivx_grids = viewers.ImageViewer3D_Sliced(ax[2,0], sdt, 0, 'Target Image - Z slice')
    warped_ivy_grids = viewers.ImageViewer3D_Sliced(ax[2,1], sdt, 1, 'Target Image - X slice')
    warped_ivz_grids = viewers.ImageViewer3D_Sliced(ax[2,2], sdt, 2, 'Target Image - Y slice')
elif SHOW_CT_IMG:
    warped_ivx_grids = viewers.ImageViewer3D_Sliced(ax[2,0], ct_img, 0, 'Target Image - Z slice')
    warped_ivy_grids = viewers.ImageViewer3D_Sliced(ax[2,1], ct_img, 1, 'Target Image - X slice')
    warped_ivz_grids = viewers.ImageViewer3D_Sliced(ax[2,2], ct_img, 2, 'Target Image - Y slice')
elif SHOW_DEMO_NPY:
    warped_ivx_grids = viewers.ImageViewer3D_Sliced(ax[2,0], image, 0, 'Target Image - Z slice')
    warped_ivy_grids = viewers.ImageViewer3D_Sliced(ax[2,1], image, 1, 'Target Image - X slice')
    warped_ivz_grids = viewers.ImageViewer3D_Sliced(ax[2,2], image, 2, 'Target Image - Y slice')
    

feh = viewers.FigureEventHandler(fig)

feh.add_axes_event('button_press_event', ax[0,0], ivx.on_mouse_press)
feh.add_axes_event('button_press_event', ax[0,1], ivy.on_mouse_press)
feh.add_axes_event('button_press_event', ax[0,2], ivz.on_mouse_press)

feh.add_axes_event('button_press_event', ax[1,0], warped_ivx.on_mouse_press)
feh.add_axes_event('button_press_event', ax[1,1], warped_ivy.on_mouse_press)
feh.add_axes_event('button_press_event', ax[1,2], warped_ivz.on_mouse_press)

if SHOW_CT_IMG or SHOW_SDT or SHOW_DEMO_NPY:
    feh.add_axes_event('button_press_event', ax[2,0], warped_ivx_grids.on_mouse_press)
    feh.add_axes_event('button_press_event', ax[2,1], warped_ivy_grids.on_mouse_press)
    feh.add_axes_event('button_press_event', ax[2,2], warped_ivz_grids.on_mouse_press)

feh.synchronize([ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2],ax[2,0], ax[2,1], ax[2,2]])

plt.show()
# plt.savefig("./data/imageViewer_double.png")