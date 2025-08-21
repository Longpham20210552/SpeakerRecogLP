import torch
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
adapter = classifier_spk(num_classes=10, feat_dim=192)
total_params = sum(p.numel() for p in adapter.parameters())
print(f"Tổng số tham số của adapter: {total_params}")

URL_TEMPLATE = "https://github.com/IDRnD/ReDimNet/releases/download/latest/{model_name}"

# Giả sử bạn đã có class Demucs và model đã load
def load_custom(model_name='b0', train_type='ptn', dataset='vox2'):
    model_name = f'{model_name}-{dataset}-{train_type}.pt'
    url = URL_TEMPLATE.format(model_name = model_name)
    full_state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']
    return state_dict
state_dict = torch.load("dns48.th", map_location="cpu")
data = torch.load("diff-sv.pt", map_location='cpu')
print(type(data))
full_state_dict = load_custom(model_name='b0', train_type='ptn', dataset='vox2')
# In danh sách tên tensor và số tham số
total_params = 0
for name, param in full_state_dict.items():
    num = param.numel()
    total_params += num
    print(f"{name:50s} | Shape: {str(tuple(param.shape)):20s} | Params: {num:,}")

print(f"\n>>> Tổng số tham số: {total_params:,}")
