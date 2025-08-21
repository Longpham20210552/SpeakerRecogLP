import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from redimnet.hubconf import ReDimNet 

model2 = ReDimNet('b0')

class Pretrained_ReDimNet(nn.Module):
    def __init__(self, model2):
        super(Pretrained_ReDimNet, self).__init__()
        self.redimnet = model2.backbone 
        self.pool = model2.pool
        self.bn = model2.bn
        self.linear = model2.linear
    def forward(self, x):                                                
        x = self.redimnet(x)
        x = self.bn(self.pool(x))
        embeddings = self.linear(x)
        return embeddings

