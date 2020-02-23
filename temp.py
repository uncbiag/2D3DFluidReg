import nibabel as nib
import numpy as np
import mermaid.module_parameters as pars
import os
import matplotlib.pyplot as plt
from CTPlayground import resample

# # Load Params
path = "./lung_registration_setting.json"
lung_reg_params = pars.ParameterDict()
lung_reg_params.load_JSON(path)

preprocessed_folder = lung_reg_params["preprocessed_folder"]
# I0_numpy = np.load(preprocessed_folder+'/I0_3d.npy')
# I1_numpy = np.load(preprocessed_folder+'/I1_3d.npy')
# I0_mask = np.load(preprocessed_folder+'/I0_mask.npy')

# I1_image = nib.Nifti1Image(I0_numpy, affine=np.eye(4))
# I1_image.header.get_xyzt_units()

# I1_image.to_filename(os.path.join('./data','I1.nii.gz')) 

# I0_image = nib.Nifti1Image(I0_numpy, affine=np.eye(4))
# I0_image.header.get_xyzt_units()
# I0_image.to_filename(os.path.join('./data','I0.nii.gz')) 

# I0_mask = I0_mask.astype(np.intc)
# I0_mask = nib.Nifti1Image(I0_mask, affine=np.eye(4))
# I0_mask.header.get_xyzt_units()
# print(I0_mask.header.get_data_dtype())
# I0_mask.to_filename(os.path.join('./data','I0_mask.nii.gz')) 

# with open('./data/testField.dat', 'rb') as f:
#     lines = f.readlines()
#     for l in lines:
#         print(l)

# data = np.fromfile('./data/testField.dat',  dtype='<f4').reshape((6,-1))
# print(data[:,0])

def convert_to_numpy(file_path, save_path):
    img = nib.load(file_path)
    img = np.array(img.dataobj)
    img = np.swapaxes(img, 0, 2)
    spacing = [1., 1., 1.]
    # new_spacing = np.array([2., 2., 2.])
    # image, spacing = resample(img, np.array(spacing), new_spacing)
    np.save(save_path, img)


# convert_to_numpy('../eval_data/copd1/copd1/prereg/warped1.nii.gz', preprocessed_folder+'/I0_prereg.npy')
# convert_to_numpy('../eval_data/copd1/copd1/prereg/moving1.nii.gz', preprocessed_folder+'/I1_prereg.npy')
convert_to_numpy('../eval_data/copd1/copd1/prereg/fixed1.nii.gz', preprocessed_folder+'/I1_temp.npy')

# data = np.fromfile('./data/points.dat',  dtype='<f4').reshape(300,3)
# np.save("./data/points.npy",data)

# plt.imshow(img[80])
# plt.show()
