import torch
import torch.nn as nn


# CNN


# 일단 구조상관없이 생성

class SIMPLEMLP_Icentia(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_Icentia, self).__init__()
        
        self.fc1 = nn.Linear(500, 600,bias=False)
        self.bn1 = nn.BatchNorm1d(600) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5) 
        self.out = nn.Linear(600, num_class,bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.out(x)
    
        
        return x
    
class SIMPLEMLP_CinC(nn.Module):
    def __init__(self, num_class=2):
        super(SIMPLEMLP_CinC, self).__init__()
        
        self.fc1 = nn.Linear(18000, 20000,bias=False)
        self.bn1 = nn.BatchNorm1d(20000) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5) 
        self.out = nn.Linear(20000, num_class,bias=False)



    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.out(x)

        return x
    




