import torch
import mermaid.utils as utils
import torch.nn.functional as F
import numpy as np
import os

import matplotlib.pyplot as plt

def readPoint(f_path):
    with open(f_path) as fp:
        content = fp.read().split('\n')

        # Read number of points from second
        count = len(content)-1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float32)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split('\t')
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])

        return points

dim = np.array([512.0, 512.0, 121.])
spacing = np.array([0.625,0.625,2.5])
source_list = readPoint("../eval_data/copd1/copd1/copd1_300_iBH_xyz_r1.txt")
source_list_t = torch.mul(torch.from_numpy(source_list), torch.from_numpy(spacing))

target_list = readPoint("../eval_data/copd1/copd1/copd1_300_eBH_xyz_r1.txt")/dim*2.0-1.0
target_list_t = torch.from_numpy(target_list).unsqueeze(0).unsqueeze(0).unsqueeze(0)
target_list_t = torch.flip(target_list_t, [4])
phi = np.load("./data/disp_svf_double.npy")
phi_t = torch.from_numpy(phi).double()

warped_list_t = F.grid_sample(phi_t, target_list_t)

warped_list_t = warped_list_t.permute(0,2,3,4,1)[0,0,0]
warped_list_t = torch.mul(warped_list_t, torch.from_numpy(spacing*dim))

pdist = torch.nn.PairwiseDistance(p=2)
dist = pdist(source_list_t, warped_list_t)
res = torch.mean(dist)
print(res)

plt.plot(dist,"o")
plt.show()
# plt.savefig("./data/eval_dir_lab.png")
