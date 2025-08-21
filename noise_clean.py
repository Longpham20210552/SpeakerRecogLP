import os
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
from denoiser.pretrained import dns48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dns48(pretrained=False)  # Không tải từ internet
state_dict = torch.load("dns48.th", map_location="cpu")
model.load_state_dict(state_dict)
model.to(device).eval()


# Cấu hình

input_root = "LP_clean_normalized/VALID"
output_root = "LP_clean_Demucs/VALID"
target_sr = 16000  # sample rate mà Demucs dùng


# Hàm xử lý 1 file

def denoise_file(input_path, output_path):
    waveform, sr = torchaudio.load(input_path)
    if sr != target_sr:
        waveform = Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    
    with torch.no_grad():
        inp = waveform.to(device)
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # (B=1, C, T)
        out = model(inp)
        out = out.squeeze(0).cpu()  # (C, T)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, out, sample_rate=target_sr)


# Duyệt folder và xử lý

for speaker in os.listdir(input_root):
    speaker_path = os.path.join(input_root, speaker)
    if not os.path.isdir(speaker_path):
        continue

    for file in tqdm(os.listdir(speaker_path), desc=f"Processing {speaker}"):
        if not file.endswith(".wav"):
            continue
        in_path = os.path.join(speaker_path, file)
        out_path = os.path.join(output_root, speaker, file.replace(".wav", "_denoised.wav"))
        denoise_file(in_path, out_path)