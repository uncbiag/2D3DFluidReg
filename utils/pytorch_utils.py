import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F


def setup_device(desired_gpu=None):
    print('Device setup:')
    print('-------------')
    if torch.cuda.is_available() and (desired_gpu is not None):
        device = torch.device('cuda:' + str(desired_gpu))
        print('Setting the default tensor type to torch.cuda.FloatTensor')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('Setting the CUDA device to {}'.format(desired_gpu))
        torch.cuda.set_device(desired_gpu)
    else:
        device = 'cpu'
        print('Setting the default tensor type to torch.FloatTensor')
        torch.set_default_tensor_type(torch.FloatTensor)
        print('Device is {}'.format(device))
    return device


class Bilinear(Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, zero_boundary=False, using_scale=True, mode="bilinear"):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__()
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.using_scale = using_scale
        self.mode = mode
        """ scale [-1,1] image intensity into [0,1], this is due to the zero boundary condition we may use here """

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]),
                                                 padding_mode=self.zero_boundary,
                                                 mode=self.mode,
                                                 align_corners=True)
        # output = torch.nn.functional.grid_sample(input1, input2.permute([0, 2, 3, 4, 1]),
        #                                          padding_mode=self.zero_boundary)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        if self.using_scale:

            output = self.forward_stn((input1 + 1) / 2, input2)
            # print(STNVal(output, ini=-1).sum())
            return output * 2 - 1
        else:
            output = self.forward_stn(input1, input2)
            # print(STNVal(output, ini=-1).sum())
            return output