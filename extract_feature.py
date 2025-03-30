import os
import torch
import pickle
import librosa
import numpy as np
from model.redimnet import MelBanks

# Chuẩn hóa tên file về lowercase
def list_files(folder_path):
    entries = os.listdir(folder_path)
    wav_files = [os.path.splitext(entry)[0] for entry in entries if entry.lower().endswith('.wav')]
    return wav_files

def normalize_frames(m, scale=True):
    if scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

# Tạo thư mục nếu chưa tồn tại
def create_subfolders(parent_folder, subfolder):
    subfolder_path = os.path.join(parent_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

# Lưu file dưới dạng pickle
def save_file_at_dir(dir_path, filename, file_content, mode='wb'):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), mode) as f:
        pickle.dump(file_content, f)

# Trích xuất đặc trưng từ audio
def extract_feature(audio_filename):
    audio, sr = librosa.load(audio_filename, sr=16000, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
    mel_banks = MelBanks(n_mels=60, hop_length=160)  
    features = mel_banks(audio_tensor).squeeze(0).numpy().T
    return features

# Lưu feature vào file .p
def save(audio, filename, onlyfilename, label):
    content = {'feat': audio, 'label': label}
    save_file_at_dir(os.path.dirname(filename), onlyfilename, content)

# ============================== #
#            MAIN               #
# ============================== #
# Đường dẫn chứa dữ liệu đầu vào
source_folder = 'TIMIT_2/TEST'
# Đường dẫn chứa dữ liệu đầu ra
target_folder = 'TIMIT_PROCESSED/TEST'

# Duyệt qua tất cả thư mục con và file trong source_folder
for root, dirs, files in os.walk(source_folder):
    for folder in dirs:
        # Tạo cấu trúc thư mục tương tự trong target_folder
        target_subfolder = os.path.join(target_folder, os.path.relpath(os.path.join(root, folder), source_folder))
        create_subfolders(target_folder, target_subfolder)
        
        source_subfolder = os.path.join(root, folder)
        subfiles = list_files(source_subfolder)

        print(f"🔎 Processing folder: {source_subfolder} -> {target_subfolder}")
        print(f"📁 Found {len(subfiles)} files")

        for file in subfiles:
            # Đường dẫn file .wav nguồn
            audio_filename = os.path.join(source_subfolder, file + ".wav")
            # Đường dẫn file .p đích
            dir_path = target_subfolder
            filename = os.path.join(dir_path, file + ".p")
            onlyfilename = file + ".p"
            label = folder

            if os.path.exists(audio_filename):
                # Trích xuất feature
                feature = extract_feature(audio_filename)
                # Lưu kết quả
                save(feature, filename, onlyfilename, label)
                print(f"✅ Saved: {filename}")
            else:
                print(f"❌ File not found: {audio_filename}")

print("🎯 Processing complete!")
