import time
import pickle
import warnings
import gc
import copy
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import noise_addition_utils
from pathlib import Path
from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from IPython.display import clear_output

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
        
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        conved = self.cconvt(x)
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
        return output
    
class DCUnet20(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """
    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.set_size(model_complexity=int(45//1.414), input_channels=1, model_depth=20)
        self.encoders = []
        self.model_length = 20 // 2
        
        for i in range(self.model_length):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1],
                             filter_size=self.enc_kernel_sizes[i], stride_size=self.enc_strides[i], padding=self.enc_paddings[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            if i != self.model_length - 1:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1], 
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i], padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i])
            else:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i], out_channels=self.dec_channels[i + 1], 
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i], padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i], last_layer=True)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)
       
        
    def forward(self, x, is_istft=True):
        # print('x : ', x.shape)
        orig_x = x
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
            # print('Encoder : ', x.shape)
            
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            # print('Decoder : ', p.shape)
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)
        # u9 - the mask
        mask = p
        # print('mask : ', mask.shape)
        output = mask * orig_x
        output = torch.squeeze(output, 1)
        # print(output.shape)
        if is_istft:
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        return output

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0)]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity,
                                 model_complexity,
                                 1]

            self.dec_kernel_sizes = [(6, 3), 
                                     (6, 3),
                                     (6, 3),
                                     (6, 4),
                                     (6, 3),
                                     (6, 4),
                                     (8, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1), #
                                (2, 2), #
                                (2, 1), #
                                (2, 2), #
                                (2, 1), #
                                (2, 2), #
                                (2, 1), #
                                (2, 2), #
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 3),
                                 (3, 0)]
            
            self.dec_output_padding = [(0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0),
                                       (0,0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))

# Hàm này không cần thiết nếu ta chỉ lấy 165000 mẫu
# def prepare_waveform(waveform, max_len=165000):
#     waveform = waveform.numpy()
#     current_len = waveform.shape[1]
#     output = np.zeros((1, max_len), dtype='float32')
#     output[0, :min(current_len, max_len)] = waveform[0, :max_len]
#     return torch.from_numpy(output)

# Các hàm prepare_and_split_waveform và prepare_and_split_waveform_overlap cũng không cần
# vì ta chỉ muốn một đoạn cố định 165000 mẫu
# def prepare_and_split_waveform(waveform, segment_len=165000):
#     """
#     Pad waveform to the nearest multiple of segment_len, then split into segments.
#     Returns: list of segments, original length
#     """
#     waveform = waveform.numpy()
#     current_len = waveform.shape[1]
#     num_segments = (current_len + segment_len - 1) // segment_len
#     padded_len = num_segments * segment_len

#     padded_waveform = np.zeros((1, padded_len), dtype='float32')
#     padded_waveform[0, :current_len] = waveform[0, :current_len]

#     # Chia thành các đoạn nhỏ
#     segments = []
#     for i in range(num_segments):
#         segment = padded_waveform[:, i * segment_len : (i + 1) * segment_len]
#         segments.append(torch.from_numpy(segment))

#     return segments, current_len

# def prepare_and_split_waveform_overlap(waveform, segment_len=165000, hop_size=82500):
#     """
#     Chia waveform thành các đoạn có chồng lấn (overlap), trả về:
#     - Danh sách các đoạn (tensor)
#     - Vị trí bắt đầu từng đoạn (list of int)
#     - Chiều dài gốc của waveform
#     """
#     waveform = waveform.numpy()
#     current_len = waveform.shape[1]
#     segments = []
#     positions = []

#     for start in range(0, current_len, hop_size):
#         end = start + segment_len
#         if end > current_len:
#             pad_len = end - current_len
#             padded = np.pad(waveform, ((0, 0), (0, pad_len)), mode='constant')
#         else:
#             padded = waveform
#         segment = padded[:, start:end]
#         segments.append(torch.from_numpy(segment.astype('float32')))
#         positions.append(start)
#         if end >= current_len:
#             break

#     return segments, positions, current_len

def run_inference_on_folder(model, input_root, output_root, n_fft=64, hop_length=16, device='cpu', target_sample_rate=48000):
    model.eval()
    input_root = Path(input_root)
    output_root = Path(output_root)

    all_files = sorted(list(input_root.rglob("*.wav")))
    
    # Kích thước đoạn cố định mà bạn muốn xử lý
    fixed_segment_len = 165000 

    for file_path in tqdm(all_files, desc="Processing"):
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Chuẩn bị đoạn âm thanh: lấy 165,000 mẫu đầu tiên hoặc đệm bằng 0
        current_len = waveform.shape[1]
        
        # Tạo một tensor trống có kích thước fixed_segment_len
        processed_waveform = torch.zeros(1, fixed_segment_len, dtype=torch.float32)

        # Sao chép dữ liệu từ waveform vào processed_waveform, chỉ lấy tối đa fixed_segment_len mẫu
        processed_waveform[0, :min(current_len, fixed_segment_len)] = waveform[0, :min(current_len, fixed_segment_len)]
        
        segment_to_process = processed_waveform.to(device)

        # Chuyển đổi sang STFT
        stft = torch.stft(segment_to_process, n_fft=n_fft, hop_length=hop_length, 
                          normalized=True, return_complex=False)
        stft = stft.unsqueeze(0).to(device) # Thêm chiều batch

        with torch.no_grad():
            output = model(stft, is_istft=False)

        # Chuyển đổi lại về miền thời gian
        output_complex = torch.view_as_complex(output.squeeze(0))
        output_waveform = torch.istft(output_complex, n_fft=n_fft, hop_length=hop_length,
                                      normalized=True, center=True)
        
        # Đảm bảo output_waveform có đúng kích thước (1, fixed_segment_len)
        if output_waveform.dim() == 1:
            output_waveform = output_waveform.unsqueeze(0)
        
        if output_waveform.shape[1] < fixed_segment_len:
            pad_size = fixed_segment_len - output_waveform.shape[1]
            output_waveform = F.pad(output_waveform, (0, pad_size))
        elif output_waveform.shape[1] > fixed_segment_len:
            output_waveform = output_waveform[:, :fixed_segment_len]
        
        output_waveform = output_waveform.cpu() # Chuyển về CPU trước khi lưu

        # Lưu file đã xử lý
        relative_path = file_path.relative_to(input_root)
        out_path = output_root / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(str(out_path), output_waveform, sample_rate=target_sample_rate)

# Hàm main lọc nhiễu 
def main():
    SAMPLE_RATE = 16000
    N_FFT = (SAMPLE_RATE * 64) // 1000  
    HOP_LENGTH = (SAMPLE_RATE * 16) // 1000  
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model weights
    model_weights_path = "Pretrained_Weights/Noise2Clean/mixed.pth"
    dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
    dcunet20.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))

    # Denoise
    input_root = "LP_process/TEST"
    output_root = "LP_process/DCUNet_n2c"

    run_inference_on_folder(dcunet20, input_root, output_root, n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE, target_sample_rate=SAMPLE_RATE)

if __name__ == "__main__":
    main()