import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def image_gradient(x, device):
    x_filter = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]])
    x_filter = x_filter.view((1,1,3,3)).to(device)
    g_x = F.conv2d(x,x_filter)

    y_filter = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]])
    y_filter = y_filter.view((1,1,3,3)).to(device)
    g_y = F.conv2d(x,y_filter)

    g = torch.sqrt(torch.pow(g_x,2)+torch.pow(g_y,2))
    return g


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

preprocessed_file_folder = '../eval_data/preprocessed/'
case_pixels = np.load(preprocessed_file_folder+'/ihale_proj.npy')
I0 = torch.from_numpy(case_pixels).unsqueeze(0)
I0 = I0.to(device)

case = np.load(preprocessed_file_folder+'/ehale_proj.npy').astype(np.float32)
I1 = torch.from_numpy(case).unsqueeze(0).to(device)

index = 4
g_I0 = image_gradient(I0[:,index:index+1,:,:], device)
g_I1 = image_gradient(I1[:,index:index+1,:,:], device)

fig, axe = plt.subplots(1,2)
axe[0].imshow(g_I0[0,0].cpu().numpy())
axe[1].imshow(g_I1[0,0].cpu().numpy())
plt.show()