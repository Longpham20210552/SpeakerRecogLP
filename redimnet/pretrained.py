import torch
import torch.nn as nn
import torch.nn.functional as F
from redimnet.model import ReDimNetWrap
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from redimnet.hubconf import ReDimNet 

class classifier_spk(nn.Module):
    def __init__(self, num_classes = 10, feat_dim = 192):
        super(classifier_spk, self).__init__()
        self.num_classes = num_classes
        # Define batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.bn1_1 = nn.BatchNorm1d(int(feat_dim*1/2))
        self.relu1_1 = nn.ReLU()
        self.fc1_1 = nn.Linear(feat_dim, int(feat_dim*1/2), bias=False)
        self.bn1_2 = nn.BatchNorm1d(int(feat_dim*1/2))
        self.relu1_2 = nn.ReLU()
        self.fc1_2 = nn.Linear(int(feat_dim*1/2), int(feat_dim*1/2), bias=False)
        self.fc2 = nn.Linear(int(feat_dim*1/2), num_classes, bias=False)
    def forward(self, x, return_feature=True):
        # Flatten the input
        x = torch.flatten(x, 1)
        # Apply the first set of layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # Apply the second set of layers
        x = self.fc1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        # Apply the third set of layers
        x = self.fc1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        # Apply the final fully connected layer to produce logits for classification
        y = self.fc2(x)
        if return_feature:
            return x, y
        else:
            return y
        
class AAMSoftmax_new(nn.Module):
    def __init__(self, nOut, nClasses, aam_margin=0.2, aam_scale=25, easy_margin=False):
        super(AAMSoftmax_new, self).__init__()
        self.m = aam_margin  # AAM margin
        self.s = aam_scale   # AAM scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        print('Initialised AAM_Softmax without labels')
    def forward(self, x):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))      # cos(theta)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1)) # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        logits = phi * self.s  
        return logits
    

model2 = ReDimNet('b0')

class ReDimNetWithClassifier(nn.Module):
    def __init__(self, model2, num_classes):
        super(ReDimNetWithClassifier, self).__init__()
        self.redimnet = model2.backbone 
        self.pool = model2.pool
        self.bn = model2.bn
        self.linear = model2.linear
        self.classifier = AAMSoftmax_new(96, nClasses = num_classes)
        self.classifier_SRPL = classifier_spk(num_classes = num_classes, feat_dim= 192)

        for param in self.redimnet.parameters():
            param.requires_grad = False
        for param in self.pool.parameters():
            param.requires_grad = False
        for param in self.bn.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False
        for param in self.classifier_SRPL.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):                                                
        x = self.redimnet(x)
        x = self.bn(self.pool(x))
        embeddings = self.linear(x)
        emb_s, _ = self.classifier_SRPL(embeddings) 
        #x = F.relu(self.bn1(embeddings)) 
        logits = self.classifier(emb_s)
        return emb_s, logits, embeddings 

# Khởi tạo mô hình ReDimNetWithClassifier
model_with_classifier = ReDimNetWithClassifier(model2, num_classes=233)
