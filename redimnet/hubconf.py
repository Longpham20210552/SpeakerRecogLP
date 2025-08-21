import os
import sys
import json
import torch
sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))
from redimnet.model import ReDimNetWrap
dependencies = ['torch','torchaudio']
URL_TEMPLATE = "https://github.com/IDRnD/ReDimNet/releases/download/latest/{model_name}"
def load_custom(model_name='b0', train_type='ptn', dataset='vox2'):
    model_name = f'{model_name}-{dataset}-{train_type}.pt'
    url = URL_TEMPLATE.format(model_name = model_name)
    full_state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']
    model = ReDimNetWrap(**model_config)
    '''
    print("\n=== TÊN VÀ KÍCH THƯỚC TRONG state_dict ===")
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")
    print("\n=== TÊN VÀ KÍCH THƯỚC TRONG MÔ HÌNH HIỆN TẠI ===")
    for k, v in model.state_dict().items():
        print(f"{k}: {v.shape}")'
    '''
    if train_type is not None:
        load_res = model.load_state_dict(state_dict)    
        print(f"load_res : {load_res}")
    return model
def ReDimNet(model_name, train_type='ptn', dataset='vox2'):
    return load_custom(model_name, train_type=train_type, dataset=dataset)