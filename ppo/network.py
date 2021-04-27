import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Might have to set up 2 networks, since relu as output activation function for actor gave weird outputs(not equal to 1)
class feedforwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(feedforwardNN, self).__init__()
        in_dim = len(in_dim)
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, np.prod(out_dim))

    def forward(self, obs):
        #Convert observation to tensor if input is numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = torch.tensor(obs, dtype=torch.float)
      
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = F.softmax(self.layer3(activation2), dim=-1)

        return output

        
        
                