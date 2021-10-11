import numpy as np
import torch
import matplotlib.pyplot as plt 

from mermaid import image_sampling as IS

smaller_npy = "../eval_data/preprocessed_smaller/Case1Pack_I0_proj.npy"
origin_npy = "../eval_data/preprocessed/Case1Pack_I0_proj.npy"

smaller_img = np.load(smaller_npy)
origin_img = torch.from_numpy(np.load(origin_npy)).unsqueeze(0)
spacing = [1., 1.]

sampler = IS.ResampleImage()
ds_origin_img, space = sampler.downsample_image_to_size(origin_img, np.array(spacing), np.array(smaller_img.shape[1:]), 3)
ds_origin_img = ds_origin_img.cpu().numpy()[0]
print(np.max(ds_origin_img-smaller_img))
im = plt.imshow((ds_origin_img-smaller_img)[2])
plt.colorbar(im)
plt.savefig("./log/dataset_analysis/diff.png")
