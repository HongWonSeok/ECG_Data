from math import sqrt

import einops
from einops.layers.torch import Reduce
import torch
import torch.nn as nn



# 일단 구조상관없이 생성

class SIMPLEMLP_Icentia(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_Icentia, self).__init__()
        
        self.fc1 = nn.Linear(500, 600,bias=False)
        self.bn1 = nn.BatchNorm1d(600) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5) 
        # self.fc2 = nn.Linear(18500, 17000,bias=False)
        # self.bn2 = nn.BatchNorm1d(17000) 
        # self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5) 
        self.out = nn.Linear(600, num_class,bias=False)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = self.relu1(x)
        x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.bn2(x)  
        # x = self.relu2(x)
        # x = self.dropout2(x)
        x = self.out(x)
    
        
        return x
    
class SIMPLEMLP_CinC(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_CinC, self).__init__()
        
        self.fc1 = nn.Linear(18000, 20000,bias=False)
        self.bn1 = nn.BatchNorm1d(20000) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5) 
        # self.fc2 = nn.Linear(18500, 17000,bias=False)
        # self.bn2 = nn.BatchNorm1d(17000) 
        # self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5) 
        self.out = nn.Linear(20000, num_class,bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = self.relu1(x)
        x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.bn2(x)  
        # x = self.relu2(x)
        # x = self.dropout2(x)
        x = self.out(x)

        return x
    

