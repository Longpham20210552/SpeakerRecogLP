import os
import pickle
import uuid

# ===== CONFIG =====
SOURCE_DIR = 'TIMIT_PROCESSED/TEST'  
OUTPUT_DIR = 'TIMIT_UNKNOWN/TEST'  
KNOWN_SPEAKERS = ['FAPB0', 'FBCH0', 'FHXS0', 'FJDM2', 'MABC0', 'MAJP0', 'MBMA1', 'MCAE0', 'MDRD0', 'MEAL0']  
UNKNOWN_SPEAKER_NAME = 'unknown'  

# ===== HÀM XỬ LÝ =====
def process_data():
    total_files = 0
    unknown_files = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith('.p'):
                source_file = os.path.join(root, file)

                try:
                    # Đọc dữ liệu từ file gốc
                    with open(source_file, 'rb') as f:
                        data = pickle.load(f)
                except Exception as e:
                    print(f"❌ Lỗi khi đọc file {source_file}: {e}")
                    continue

                # Lấy thông tin người nói từ đường dẫn
                speaker = os.path.basename(os.path.dirname(source_file))
                dataset = os.path.basename(os.path.dirname(os.path.dirname(source_file)))

                # ✅ Xác định nhãn và thư mục đích
                if speaker in KNOWN_SPEAKERS:
                    target_dir = os.path.join(OUTPUT_DIR, dataset, speaker)
                else:
                    target_dir = os.path.join(OUTPUT_DIR, dataset, UNKNOWN_SPEAKER_NAME)
                    data['label'] = UNKNOWN_SPEAKER_NAME
                    unknown_files += 1

                # ✅ Tạo thư mục đích nếu chưa tồn tại
                os.makedirs(target_dir, exist_ok=True)

                # ✅ Thêm UUID vào tên file để tránh ghi đè
                unique_id = str(uuid.uuid4())[:8]
                new_filename = f"{unique_id}_{file}"
                target_file = os.path.join(target_dir, new_filename)

                try:
                    with open(target_file, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"✅ Đã ghi file: {target_file}")
                except Exception as e:
                    print(f"❌ Lỗi khi ghi file {target_file}: {e}")
                    continue

                total_files += 1

    print(f"✅ Đã xử lý tổng cộng {total_files} file.")
    print(f"🔄 Đã gán nhãn 'unknown' cho {unknown_files} file.")

# ===== CHẠY HÀM =====
if __name__ == "__main__":
    process_data()
