import os
import shutil
import random

# Đường dẫn thư mục gốc chứa dữ liệu
SOURCE_DIR = 'TRAIN_SOURCE'
TRAIN_DIR = 'TIMIT_2/TRAIN'
VALID_DIR = 'TIMIT_2/VALID'
TEST_DIR = 'TIMIT_2/TEST'

# Đặt seed để kết quả ngẫu nhiên có thể lặp lại
random.seed(42)

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data():
    # Đọc danh sách folder trong thư mục gốc
    speakers = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    print(speakers) 
    for speaker in speakers:
        source_speaker_path = os.path.join(SOURCE_DIR, speaker)
        train_speaker_path = os.path.join(TRAIN_DIR, speaker)
        test_speaker_path = os.path.join(TEST_DIR, speaker)
        valid_speaker_path = os.path.join(VALID_DIR, speaker)
        
        # Tạo folder đích cho mỗi người nói trong TRAIN và TEST
        create_dir_if_not_exists(train_speaker_path)
        create_dir_if_not_exists(test_speaker_path)
        create_dir_if_not_exists(valid_speaker_path)
        
        # Lọc ra các file .wav trong thư mục
        wav_files = [f for f in os.listdir(source_speaker_path) if f.endswith('.WAV')]
        
        # Chỉ lấy đúng 10 file nếu có đủ
        if len(wav_files) >= 10:
            selected_files = random.sample(wav_files, 10)  # Chọn ngẫu nhiên 10 file
            
            # Chia thành 8 file cho TRAIN và 2 file cho TEST
            train_files = selected_files[:7]
            valid_files = selected_files[7:8]
            test_files = selected_files[8:]
            
            # Copy file vào thư mục TRAIN
            for file in train_files:
                shutil.copy(os.path.join(source_speaker_path, file), os.path.join(train_speaker_path, file))
            
            # Copy file vào thư mục TEST
            for file in test_files:
                shutil.copy(os.path.join(source_speaker_path, file), os.path.join(test_speaker_path, file))
            for file in valid_files:
                shutil.copy(os.path.join(source_speaker_path, file), os.path.join(valid_speaker_path, file))
            
            print(f"✔️ {speaker}: {len(train_files)} files in TRAIN, {len(test_files)} files in TEST, {len(valid_files)} files in VALID")
        else:
            print(f"❌ {speaker}: Not enough files ({len(wav_files)} files found)")

if __name__ == "__main__":
    split_data()
