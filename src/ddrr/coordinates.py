import torch
from typing import List
import math
from math import cos, sin

def compose_transformation(translation: List[float], alpha: float, beta: float, gamma: float, scale: List[float] = [1.,1.,1.]):
    alpha = alpha/180.*math.pi
    beta = beta/180.*math.pi
    gamma = gamma/180.*math.pi
    r_x = torch.tensor([[1.,0.,0.,0.], [0., cos(alpha), -sin(alpha), 0.], [0., sin(alpha), cos(alpha), 0.], [0.,0.,0.,1.]])
    r_y = torch.tensor([[cos(beta), 0., sin(beta), 0.], [0., 1., 0., 0.], [-sin(beta), 0., cos(beta), 0.], [0.,0.,0.,1.]])
    r_z = torch.tensor([[cos(gamma), -sin(gamma), 0., 0.], [sin(gamma), cos(gamma), 0., 0.], [0., 0., 1., 0.], [0.,0.,0.,1.]])
    translation = torch.tensor([[1.,0.,0.,translation[0]], [0.,1.,0.,translation[1]], [0.,0.,1.,translation[2]], [0.,0.,0.,1.]])
    scale = torch.tensor([[scale[0],0.,0.,0.], [0.,scale[1],0.,0.], [0.,0.,scale[2],0.], [0.,0.,0.,1.]])
    return translation@r_z@r_y@r_x@scale

def to_homo(p):
    shape = p.shape[:-1] + (1,)
    return torch.cat([p, torch.ones(shape, device=p.device)], dim=-1)


class Coordinate():
    def __init__(self, translation: List[float]=[0.,0.,0.], rotation: List[float]=[0.,0.,0.], scale: List[float]=[1.,1.,1.]):
        transform = compose_transformation(translation, rotation[0], rotation[1], rotation[2], scale)
        self.transform = transform.unsqueeze(0)
        self.inv_transform = torch.linalg.inv(self.transform)

    def to_world(self, pos):
        return torch.matmul(self.transform, pos.unsqueeze(-1)).squeeze(-1)
    
    def to_local(self, pos):
        return torch.matmul(self.inv_transform, pos.unsqueeze(-1)).squeeze(-1)

class CarmCoordinate():
    def __init__(self, 
        translation: List[float]=[0.,0.,0.], 
        rotation: List[float]=[0.,0.,0.], 
        arms: List[List[List[float]]]=[[[0.,0.,0.],[0.,0.,0.]]],
        ):
        self.device_transform = compose_transformation(translation, rotation[0], rotation[1], rotation[2]).unsqueeze(0)
        self.arms_transform = torch.cat(
                [compose_transformation(arm[0], arm[1][0], arm[1][1], arm[1][2]).unsqueeze(0) for arm in arms],
                dim=0
            )
        self.transform = self.device_transform@self.arms_transform
        self.inv_transform = torch.linalg.inv(self.transform)

    def to_world(self, pos):
        return torch.matmul(self.transform.unsqueeze(1), pos.unsqueeze(-1)).squeeze(-1)
    
    def to_local(self, pos):
        return torch.matmul(self.inv_transform.unsqueeze(1), pos.unsqueeze(-1)).squeeze(-1)

if __name__ == "__main__":
    patient = Coordinate([0.,0.,0.], [10., 0., 0.])
    carms = CarmCoordinate([0.,0.,0.], [20., 0., 0.], arms=[[[0., 0., 0.,], [20., 0., 0.]], [[0., 0., 0.,], [40., 0., 0.]]]) 

    points_in_carms = to_homo(torch.rand(2, 20, 3))
    points_in_patient = patient.to_local(carms.to_world(points_in_carms).reshape(-1, 4))
    points_in_carms_back = carms.to_local(patient.to_world(points_in_patient).reshape(2, 20, 4))
    print(torch.sum((points_in_carms-points_in_carms)**2))
    
