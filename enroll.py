import torch
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c

from DB_wav_reader import read_feats_structure_train
from SR_Dataset import read_MFB, ToTensorTestInput
from redimnet.pretrained import ReDimNetWithClassifier
from redimnet.pretrained_model import Pretrained_ReDimNet
from redimnet.model import ReDimNetWrap 
from redimnet.hubconf import ReDimNet

# Hàm tải mô hình đã huấn luyện
# Tham số:
#   use_cuda: True nếu muốn sử dụng GPU
#   log_dir: Thư mục chứa mô hình đã lưu
#   cp_num: Số thứ tự checkpoint cần tải (hiện tại chưa thấy sử dụng, có thể cần chỉnh thêm)
#   embedding_size: Kích thước vector nhúng (chưa truyền vào model, nên tham số này chưa dùng)
#   n_classes: Số lượng lớp phân loại (chưa truyền vào model, nên tham số này chưa dùng)
def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    backbone = ReDimNet('b0')  # Khởi tạo backbone ReDimNet
    model = Pretrained_ReDimNet(backbone)  # Bọc backbone vào mô hình có thêm classifier
    if use_cuda:
        model.cuda()  # Chuyển model sang GPU nếu cần
    print('=> loading checkpoint')
    model.eval()  # Đặt chế độ đánh giá (không update trọng số)
    return model

# Hàm tách tập enroll và test từ cấu trúc dữ liệu
def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure_train(dataroot_dir)  # Đọc toàn bộ cấu trúc file
    enroll_DB = DB_all[DB_all['filename'].str.contains('enroll.p')]  # Lọc file enroll
    test_DB = DB_all[DB_all['filename'].str.contains('test.p')]  # Lọc file test
    
    # Đặt lại chỉ số index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB

# Hàm trích xuất vector nhúng từ 1 file đặc trưng .p
# test_frames: số lượng frame sẽ lấy cho mỗi đoạn (cắt đoạn nếu file dài)
def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename)  # Đọc đặc trưng từ file .p
    print (input.shape)
    
    # Tính số đoạn cần cắt ra
    tot_segments = math.ceil(len(input) / test_frames)  
    
    activation = 0  # Khởi tạo tổng embedding
    
    with torch.no_grad():  # Không tính gradient
        for i in range(tot_segments):
            # Cắt đoạn hiện tại
            temp_input = input[i * test_frames : i * test_frames + test_frames]
            print(temp_input.shape)
            
            # Chuyển thành tensor phù hợp đầu vào mạng
            TT = ToTensorTestInput()
            temp_input = TT(temp_input)  # size: (1, 1, n_dims, n_frames)
            
            if use_cuda:
                temp_input = temp_input.cuda()
                
            temp_activation = model(temp_input)  # Trích xuất embedding cho đoạn
            activation += torch.sum(temp_activation, dim=0, keepdim=True)  # Cộng dồn

    print(activation.shape)
    activation = l2_norm(activation, 1)  # Chuẩn hóa L2
    print(activation.shape)          
    return activation

# Hàm chuẩn hóa L2 cho vector nhúng
def l2_norm(input, alpha):
    input_size = input.size()  
    buffer = torch.pow(input, 2)  
    normp = torch.sum(buffer, 1).add_(1e-10)  # Cộng epsilon tránh chia cho 0
    norm = torch.sqrt(normp)  
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))  # Chuẩn hóa từng vector
    output = _output.view(input_size)
    
    output = output * alpha  # Nhân thêm hệ số alpha như trong paper
    return output

# Hàm enroll tự động theo cấu trúc thư mục (mỗi thư mục con là 1 speaker)
def enroll_per_spk_from_folder(use_cuda, test_frames, model, root_dir, embedding_dir):
    """
    root_dir: thư mục chứa các thư mục con ứng với từng speaker
    embedding_dir: nơi lưu các vector nhúng trung bình của từng speaker
    """
    embeddings = {}
    speaker_list = sorted(os.listdir(root_dir))
    
    print("Start to aggregate embeddings from directory structure")
    
    for spk in speaker_list:
        spk_dir = os.path.join(root_dir, spk)
        if not os.path.isdir(spk_dir):
            continue  # Bỏ qua nếu không phải thư mục

        total_embedding = 0
        count = 0

        # Lặp qua từng file đặc trưng trong thư mục
        for file in os.listdir(spk_dir):
            if file.endswith('.p'):
                file_path = os.path.join(spk_dir, file)
                emb = get_embeddings(use_cuda, file_path, model, test_frames)
                total_embedding += emb
                count += 1

        if count > 0:
            avg_embedding = total_embedding / count  # Trung bình các embedding
            embeddings[spk] = avg_embedding

            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)
            
            embedding_path = os.path.join(embedding_dir, spk + '.pth')
            torch.save(avg_embedding, embedding_path)  # Lưu file nhúng
            print(f"Saved embedding for {spk}, {count} files aggregated.")
        else:
            print(f"No .p files found for {spk}, skipping.")

    return embeddings

# Hàm enroll với đầu vào là DataFrame thay vì cấu trúc thư mục
def enroll_per_spk(use_cuda, test_frames, model, DB, embedding_dir):
    """
    Đầu ra là dictionary lưu vector nhúng trung bình cho từng speaker
    """
    n_files = len(DB) 
    enroll_speaker_list = sorted(set(DB['speaker_id']))  # Danh sách speaker duy nhất
    
    embeddings = {}
    
    print("Start to aggregate all the d-vectors per enroll speaker")
    
    for i in range(n_files):
        filename = DB['filename'][i]
        spk = DB['speaker_id'][i]
        
        activation = get_embeddings(use_cuda, filename, model, test_frames)
        
        if spk in embeddings:
            embeddings[spk] += activation
        else:
            embeddings[spk] = activation
            
        print(f"Aggregates the activation (spk: {spk})")
        
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
        
    for spk_index in enroll_speaker_list:
        embedding_path = os.path.join(embedding_dir, spk_index + '.pth')
        torch.save(embeddings[spk_index], embedding_path)
        print(f"Save the embeddings for {spk_index}")
        
    return embeddings

# Hàm main
def main():
        
    # Cấu hình
    use_cuda = True
    log_dir = 'model_saved'
    embedding_size = 128
    cp_num = 30  # Số checkpoint cần dùng (chưa áp dụng trong hàm load_model hiện tại)
    n_classes = 5
    test_frames = 400  # Số frame mỗi đoạn test
    
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    
    root_enroll_dir = 'TIMIT_UNKNOWN/TRAIN'  # Thư mục chứa dữ liệu enroll
    embedding_dir = 'embedding_TIMIT'  # Nơi lưu kết quả vector nhúng

    enroll_per_spk_from_folder(use_cuda, test_frames, model, root_enroll_dir, embedding_dir)

if __name__ == '__main__':
    main()
