import torch
from torch import tensor
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import floatify_obs

class feedforwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):

        super(feedforwardNN, self).__init__()
        self.out_dim = out_dim
        self.in_dim = len(in_dim)
        self.layer1 = nn.Linear(self.in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, self.out_dim)

    def forward(self, obs):
        if self.out_dim == 1:
            activation1 = F.relu(self.layer1(obs))
            activation2 = F.relu(self.layer2(activation1))
            output = self.layer3(activation2)
        else:
            activation1 = F.tanh(self.layer1(obs))
            activation2 = F.tanh(self.layer2(activation1))
            output = F.softmax(self.layer3(activation2), dim=-1)

        return output
        
        
                