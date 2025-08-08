import torch
import torch.nn as nn
import math


class HuEtAl(nn.Module):

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels) 
            x = self.pool(self.conv(x)) 
        return x.numel() 

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            kernel_size = math.ceil(input_channels / 9)）
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)
        
        self.input_channels = input_channels

        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x): 
        x = x.unsqueeze(1)
        x = self.conv(x) 
        x = torch.tanh(x) 
        x = self.pool(x)
        x = torch.tanh(x) 
        x = x.view(-1, self.features_size) 
        x = self.fc1(x) 
        x = torch.tanh(x) 
        # x = self.fc2(x) 
        return x


class Multi_Branch(nn.Module):
    def __init__(self, x_channels, cA_channels, cD_channels, dim, n_classes):
        super(Multi_Branch, self).__init__()
        self.cA = HuEtAl(cA_channels, n_classes)
        self.total = HuEtAl(200, n_classes) 
        self.fc12 = nn.Linear(200, 100)
        self.fc22 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, cA, cD, cA1, cD1):
        output_cA = self.cA(cA)
        output_cD = self.cA(cD)
        output_cA1 = self.cA(cA1)
        output_cD1 = self.cA(cD1)
        concatenate = torch.cat([output_cA, output_cD1], axis=1)
        concatenate1 = torch.cat([output_cA1, output_cD], axis=1)
  
        concatenate = self.fc12(concatenate)
        concatenate1 = self.fc22(concatenate1)
        concatenate2 = torch.cat([concatenate, concatenate1], axis=1)
        concatenate2 = nn.ReLU(self.total(concatenate2))
        output = self.fc2(concatenate2)
        return output