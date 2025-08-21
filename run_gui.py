import os 
import numpy as np
import librosa
import RPi.GPIO as GPIO 
from scipy.io.wavfile import write
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as transforms
import time
import tkinter as tk
from tkinter import messagebox

# === GPIO Setup ===
RELAY_PIN = 17  
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  

# === Preprocessing Modules ===
class TruncatedInputfromMFB_NotRandom(object):
    def __init__(self, input_per_file=1):
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)

        win_size = 400
        half_win_size = int(win_size / 2)

        while num_frames - half_win_size <= half_win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)

        center_index = num_frames // 2
        for _ in range(self.input_per_file):
            start_index = max(0, center_index - half_win_size)
            end_index = start_index + win_size
            frames_slice = frames_features[start_index:end_index]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)

class ToTensorInput(object):
    def __call__(self, np_feature):
        if isinstance(np_feature, np.ndarray):
            return torch.from_numpy(np_feature.transpose((0, 2, 1))).float()

class NormalizeAudio(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True, unbiased=False) + self.eps)

class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10), freq_start_bin=0):
        super().__init__()
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        D = fea if dim == 1 else time
        width_range = self.freq_mask_width if dim == 1 else self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) & (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)
        mask = mask.unsqueeze(2 if dim == 1 else 1)

        return x.masked_fill_(mask, 0.0).view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class MelBanks(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min=20, 
        f_max=7600, 
        n_mels=80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        spec_norm='mn',
        freq_start_bin=0,
        num_apply_spec_aug=1,
        freq_mask_width=(0, 8), 
        time_mask_width=(0, 10),
    ):
        super().__init__()
        self.num_apply_spec_aug = num_apply_spec_aug
        self.torchfbank = nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),
            PreEmphasis() if do_preemph else nn.Identity(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                f_min=f_min, f_max=f_max, n_mels=n_mels, window_fn=torch.hamming_window, power=2.0)
        )

        if spec_norm == 'mn':
            self.spec_norm = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        elif spec_norm == 'mvn':
            self.spec_norm = lambda x: (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-8)
        elif spec_norm == 'bn':
            self.spec_norm = nn.BatchNorm1d(n_mels)
        else:
            self.spec_norm = nn.Identity()

        self.specaug = FbankAug(freq_start_bin=freq_start_bin,
                                freq_mask_width=freq_mask_width,
                                time_mask_width=time_mask_width) if do_spec_aug else nn.Identity()

    def forward(self, x):
        x = x.float()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = self.spec_norm(x)
            if self.training:
                for _ in range(self.num_apply_spec_aug):
                    x = self.specaug(x)
        return x

# === Audio Recording ===
def record_audio():
    print("[Recording via arecord...]")
    start = time.time()
    os.system("arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -t wav -d 3 recorded.wav")
    end = time.time()
    print(f"[Recording done] Time taken: {end - start:.3f} seconds")

# === Feature Extraction ===
def extract_feature(audio_filename):
    audio, _ = librosa.load(audio_filename, sr=16000, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    mel_banks = MelBanks(n_mels=60, hop_length=160)
    features = mel_banks(audio_tensor).squeeze(0).numpy().T
    return features

# === ONNX Classifier ===
class ONNXSpeakerClassifier:
    def __init__(self, onnx_path, speaker_list=None, threshold=0.9):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.threshold = threshold
        self.speakers = speaker_list if speaker_list else [f"spk{i}" for i in range(100)]

    def classify(self, input_tensor):
        logits = self.session.run([self.output_name], {self.input_name: input_tensor.astype(np.float32)})[0]
        probs = torch.softmax(torch.tensor(logits), dim=1)
        max_val, max_idx = torch.max(probs, dim=1)
        return probs.squeeze(0).tolist(), max_val.item(), max_idx.item()

# === GUI Application ===
def run_gui():
    onnx_model_path = 'speaker_model.onnx'
    speaker_names = ['Nguyen_Dinh_Minh', 'Nguyen_Huu_Trung', 'Nguyen_Thi_Cam_Ly', 'Pham_Quy_Long', 'Tran_Anh_Huy']
    model = ONNXSpeakerClassifier(onnx_model_path, speaker_list=speaker_names)

    root = tk.Tk()
    root.title("Speaker Recognition")
    root.geometry("400x300")

    result_text = tk.StringVar()
    result_text.set("No result yet.")

    # Define function INSIDE run_gui and pass model as default argument
    def do_record_and_recognition(model=model):
        record_audio()

        recog_start = time.time()  # Record timestamp after audio recording

        features = extract_feature("recorded.wav")
        processor = transforms.Compose([
            TruncatedInputfromMFB_NotRandom(),
            ToTensorInput()
        ])
        x = processor(features).unsqueeze(1).numpy()

        probs, max_val, max_idx = model.classify(x)

        recog_end = time.time()  # After classification

        result_lines = [f"{spk:25s}: {score:.4f}" for spk, score in zip(model.speakers, probs)]
        result_string = "\n".join(result_lines)
        print("\n=== Scores per Class ===\n" + result_string)

        if max_val > model.threshold:
            time_to_unlock = recog_end - recog_start
            print(f"[INFO] Time from end of recording to door unlock: {time_to_unlock:.3f} seconds")

            result_text.set(f"Speaker: {model.speakers[max_idx]}\nDoor unlocked after: {time_to_unlock:.3f} seconds")
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            time.sleep(5)
            GPIO.output(RELAY_PIN, GPIO.LOW)
        else:
            result_text.set("Speaker: Unknown")
            GPIO.output(RELAY_PIN, GPIO.LOW)

        messagebox.showinfo("Done", "Recognition completed.")

    tk.Label(root, text="Speaker Recognition Demo", font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="Record Audio", command=do_record_and_recognition, width=25).pack(pady=20)
    tk.Label(root, textvariable=result_text, fg="blue", font=("Arial", 12), wraplength=350).pack(pady=20)

    root.mainloop()


if __name__ == '__main__':
    run_gui()