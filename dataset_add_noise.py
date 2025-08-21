import os
import random
import soundfile as sf
import numpy as np

# Đường dẫn đến thư mục dữ liệu
DATA_DIR = 'LP_noise'
NOISE_DIR = 'free-sound'  # Thư mục chứa file nhiễu
TARGET_SNR_LIST = [-5, 0, 5, 10, 15, 20]  # 6 mức SNR: 0dB và 5dB

def load_noise_files(noise_dir):
    noise_files = []
    for root, _, files in os.walk(noise_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                noise_files.append(os.path.join(root, file))
    return noise_files

def add_noise_to_audio(audio, noise, snr_db):
    # Đảm bảo noise và audio là mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1)

    # Đảm bảo noise có cùng độ dài với audio
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    else:
        noise = np.pad(noise, (0, len(audio) - len(noise)), mode='wrap')

    # Tính năng lượng tín hiệu và noise
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)

    # Tính hệ số tỉ lệ theo SNR
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / (noise_power + 1e-8))
    noise = noise * scaling_factor

    # Thêm noise vào audio
    noisy_audio = audio + noise
    return noisy_audio

def process_dataset(split):
    print(f"Đang xử lý tập {split}...")

    input_dir = os.path.join(DATA_DIR, split)
    noise_files = load_noise_files(NOISE_DIR)

    if not noise_files:
        print("Không tìm thấy file nhiễu nào")
        return

    speakers = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for speaker in speakers:
        speaker_dir = os.path.join(input_dir, speaker)
        files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]

        for file in files:
            file_path = os.path.join(speaker_dir, file)
            audio, sr = sf.read(file_path)

            # Chuyển stereo -> mono nếu cần
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Lưu lại file gốc (không đổi)
            sf.write(file_path, audio, sr)

            for snr_db in TARGET_SNR_LIST:
                noise_file = random.choice(noise_files)
                noise, _ = sf.read(noise_file)

                # Thêm nhiễu vào tín hiệu
                noisy_audio = add_noise_to_audio(audio, noise, snr_db)

                # Đặt tên file mới
                base_name = os.path.splitext(file)[0]
                noisy_file_name = f"{base_name}_{snr_db}db.wav"
                noisy_file_path = os.path.join(speaker_dir, noisy_file_name)

                # Lưu file
                sf.write(noisy_file_path, noisy_audio, sr)
                print(f"Đã tạo: {noisy_file_name}")

def count_files(split):
    input_dir = os.path.join(DATA_DIR, split)
    speakers = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    total_files = 0
    for speaker in speakers:
        speaker_dir = os.path.join(input_dir, speaker)
        files = [f for f in os.listdir(speaker_dir) if f.lower().endswith('.wav')]
        print(f"{speaker}: {len(files)} files")
        total_files += len(files)

    print(f"Tổng số file trong tập {split}: {total_files}")

# Chạy cho cả 3 tập
process_dataset('TRAIN')
process_dataset('TEST')
process_dataset('VALID')

# Thống kê số lượng file
count_files('TRAIN')
count_files('TEST')
count_files('VALID')
