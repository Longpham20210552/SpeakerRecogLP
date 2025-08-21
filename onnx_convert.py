import torch, os, torch.nn as nn 
from redimnet.pretrained import ReDimNetWithClassifier 
from redimnet.hubconf import ReDimNet 
from SRPL import ARPLoss

# Định nghĩa mô hình nhận dạng người nói đã huấn luyện, load trọng số
class SpeakerInferenceModel(nn.Module):
    def __init__(self, model_path, feat_dim=96, n_classes=5):
        super().__init__()
        backbone = ReDimNet('b0')
        self.model = ReDimNetWithClassifier(backbone, num_classes=n_classes)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict = False)
        self.model.eval()

        self.criterion = ARPLoss(
            use_gpu=False,
            weight_pl=0.1,
            feat_dim=feat_dim,
            temp=1,
            num_classes=n_classes,
        )
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.criterion.eval()
        self.speaker_fold = os.listdir('LP_clean_unknown_split1/TRAIN')

    def forward(self, x):
        with torch.no_grad():
            print(x.shape)
            embeddings, output,_ = self.model(x)
            print(embeddings.shape)
            logits, _ = self.criterion(embeddings)
            return logits, embeddings
# Đóng gói mô hình         
class ONNXExportWrapper(nn.Module):
    def __init__(self, speaker_model):
        super().__init__()
        self.model = speaker_model.model
        self.criterion = speaker_model.criterion

    def forward(self, x):
        embeddings, _, _ = self.model(x)
        logits, _ = self.criterion(embeddings)
        return logits, embeddings
    
# Load mô hình đã huấn luyện
model_path = 'model_save_split_1/checkpoint_epoch_' + str(50) + '.pth'
speaker_model = SpeakerInferenceModel(model_path)
# Gói lại để dễ export
model_for_export = ONNXExportWrapper(speaker_model)
model_for_export.eval()

dummy_input = torch.randn(1, 1, 60, 400)

# Export
torch.onnx.export(
    model_for_export,                  # model
    dummy_input,                       # dummy input
    "speaker_model.onnx",             # output path
    export_params=True,
    opset_version=11,                  # phổ biến và đủ dùng
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits', 'embeddings'],
    dynamic_axes={
        'input': {2: 'time'},         # thời gian có thể thay đổi
        'logits': {0: 'batch'},
        'embeddings': {0: 'batch'}
    }
)