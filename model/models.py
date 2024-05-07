from math import sqrt

import einops
from einops.layers.torch import Reduce
import torch
import torch.nn as nn



# 일단 구조상관없이 생성


class SIMPLEMLP_MIT(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_MIT, self).__init__()
        
        self.fc = nn.Linear(187, 256,bias=False)
        self.bn = nn.BatchNorm1d(256) 
        self.out = nn.Linear(256, num_class,bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)  
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
    
        
        return x

class SIMPLEMLP_Icentia(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_Icentia, self).__init__()
        
        self.fc = nn.Linear(500, 600,bias=False)
        self.bn = nn.BatchNorm1d(600) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.out = nn.Linear(600, num_class,bias=False)


    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)  
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
    
        
        return x
    
class SIMPLEMLP_CinC(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_CinC, self).__init__()
        
        self.fc = nn.Linear(18000, 20000,bias=False)
        self.bn = nn.BatchNorm1d(20000) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.out = nn.Linear(20000, num_class,bias=False)



    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)  
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
    

