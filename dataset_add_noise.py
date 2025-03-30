import os
import random
import soundfile as sf
import numpy as np

# Đường dẫn đến thư mục dữ liệu
DATA_DIR = 'TIMIT_2'
NOISE_DIR = 'free-sound'  # Thư mục chứa file nhiễu
TARGET_SNR_DB = 15  # Đặt SNR ở mức 10dB

def load_noise_files(noise_dir):
    noise_files = []
    for root, _, files in os.walk(noise_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                noise_files.append(os.path.join(root, file))
    return noise_files

def add_noise_to_audio(audio, noise, snr_db):
    # Đảm bảo độ dài của noise bằng với độ dài của audio
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    else:
        noise = np.pad(noise, (0, len(audio) - len(noise)), mode='wrap')

    # Tính toán năng lượng của tín hiệu và noise
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Tính tỉ lệ SNR
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / (noise_power + 1e-8))
    noise = noise * scaling_factor

    # Thêm noise vào audio
    noisy_audio = audio + noise
    return noisy_audio

def process_dataset(split):
    print(f"🔄 Đang xử lý tập {split}...")

    input_dir = os.path.join(DATA_DIR, split)
    noise_files = load_noise_files(NOISE_DIR)

    if not noise_files:
        print("❌ Không tìm thấy file nhiễu nào!")
        return
    
    speakers = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for speaker in speakers:
        speaker_dir = os.path.join(input_dir, speaker)
        files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]

        for file in files:
            file_path = os.path.join(speaker_dir, file)
            audio, sr = sf.read(file_path)

            # ✅ Giữ nguyên file gốc
            sf.write(file_path, audio, sr)

            # ✅ Thêm 2 biến thể nhiễu cho mỗi file
            for i in range(2):
                noise_file = random.choice(noise_files)
                noise, _ = sf.read(noise_file)

                # Thêm nhiễu vào audio
                noisy_audio = add_noise_to_audio(audio, noise, TARGET_SNR_DB)

                # Lưu file có nhiễu
                noise_file_name = f"{os.path.splitext(file)[0]}_noise{i + 1}.wav"
                noise_file_path = os.path.join(speaker_dir, noise_file_name)
                sf.write(noise_file_path, noisy_audio, sr)

                print(f"✅ Đã thêm nhiễu {i + 1} cho: {noise_file_path}")

def count_files(split):
    input_dir = os.path.join(DATA_DIR, split)
    speakers = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_files = 0
    for speaker in speakers:
        speaker_dir = os.path.join(input_dir, speaker)
        files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]
        print(f"📂 {speaker}: {len(files)} files")
        total_files += len(files)

    print(f"🔥 Tổng số file trong tập {split}: {total_files}")

# Xử lý tập TRAIN và TEST
process_dataset('TRAIN')
process_dataset('TEST')
process_dataset('VALID')

# Kiểm tra số lượng file sau khi thêm nhiễu
count_files('TRAIN')
count_files('TEST')
count_files('VALID')
