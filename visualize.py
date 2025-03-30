import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
import configure as c
import json
from pathlib import Path 
import torchvision.transforms as transforms
from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import background_resnet
from redimnet.pretrained import ReDimNetWithClassifier 
from redimnet.model import ReDimNetWrap 
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, TruncatedInputfromMFB_NotRandom
from redimnet.hubconf import ReDimNet 
from SRPL import ARPLoss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
# Tải mô hình đã được lưu trữ vào chế độ training
# Tham số : use_cuda: Dùng GPU hay k
#           log_dir: Đường dẫn tới các tệp checkpoint của mô hình
#           cp_num: Số thứ tự checkpoint sẽ được tải
#           embedding_size: Kích thước vector nhúng muốn trích xuất
#           n_classes: Số lượng đầu ra cần dự đoán (số người tham gia train)
def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    backbone = ReDimNet('b0')
    #backbone = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=False)
    model = ReDimNetWithClassifier(backbone, num_classes = n_classes)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    checkpoint = torch.load('model_saved_SRPL_580/checkpoint_epoch_' + str(100) + '.pth')
    # Nạp tham số vào mô hình
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Hàm chia file enroll và test
def split_enroll_and_test(dataroot_dir):
    DB_all = read_feats_structure(dataroot_dir)
    enroll_DB = pd.DataFrame()
    test_DB = pd.DataFrame()
    
    enroll_DB = DB_all[DB_all['type_id'].str.contains('enroll')]
    test_DB = DB_all[DB_all['type_id'].str.contains('test')]
    
    # Reset the index
    enroll_DB = enroll_DB.reset_index(drop=True)
    test_DB = test_DB.reset_index(drop=True)
    return enroll_DB, test_DB


def load_enroll_embeddings(embedding_dir):
    embeddings = {}
    for f in os.listdir(embedding_dir):
        spk = f.replace('.pth','')
        # Select the speakers who are in the 'enroll_spk_list'
        embedding_path = os.path.join(embedding_dir, f)
        tmp_embeddings = torch.load(embedding_path)
        embeddings[spk] = tmp_embeddings
        
    return embeddings

def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename) # input size:(n_frames, n_dims)
    tot_segments = math.ceil(len(input)/test_frames) # Chia thành nhiều đoạn, mỗi đoạn có số frame = test frames, tổng có tot_segments đoạn
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames] 
            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # chuyển kích thước từ 2d sang 4d để đưa vào model:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True) #Cộng tất cả các đầu ra model để gộp lại
    
    activation = l2_norm(activation, 1) 
                
    return activation

def l2_norm(input, alpha):
    input_size = input.size()  # kích thước :(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. kích thước :(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output

# Toàn bộ phần trên giống file enroll.py, muốn xem thêm về công dụng của từng phần xem ở file enroll.py

def perform_identification(use_cuda, model, embeddings, test_path, test_frames, spk_list, test_speaker):
    '''
    directory = test_path
    test_embedding = None 
    dem = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            test_filename = os.path.join(root, filename)
            if dem == 0:
                test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames) # Trích xuất d-vector của người test
            else: 
                test_embedding += get_embeddings(use_cuda, test_filename, model, test_frames)
            dem +=1
    '''
    test_embedding = get_embeddings(use_cuda, test_path, model, test_frames)
    #print(test_embedding.shape) # Kiểm tra kích thước d-vector
    max_score = -10**8          # Điểm ban đầu = âm vô cùng
    best_spk = None
    for spk in spk_list:
        score = F.cosine_similarity(test_embedding, embeddings[spk]) # Tính điểm hệ số cosine giữa 2 người
        score = score.data.cpu().numpy()                             # Chuyển điểm về cpu
        if score > max_score:
            max_score = score                                        # So sánh điểm
            best_spk = spk
    if max_score > 0.0:
        #print("Speaker identification result : %s" %best_spk)
        #print(max_score)
        #true_spk = test_filename.split('/')[-2].split('_')[0]
        #print("True speaker : %s\nPredicted speaker : %s\n" %(test_speaker, best_spk,))
        return best_spk
    else: 
        best_spk = "unknown"
        #print("Speaker identification result : Không nhận dạng được")
        #print(max_score)
        #true_spk = test_filename.split('/')[-2].split('_')[0]
        #print("True speaker : %s\nPredicted speaker : %s\n" %(test_speaker, best_spk,))
        #print("True speaker : %s\nPredicted speaker : Không nhận dạng được\n" %(test_speaker,))
        return best_spk

def perform_classification(use_cuda, test_path, model, test_frames, threshold = 0.85):
    train_data = 'TIMIT_UNKNOWN/TRAIN'
    checkpoint = torch.load('model_saved_SRPL_580/checkpoint_epoch_100.pth')
    speaker_fold = os.listdir(train_data)
    criterion = ARPLoss(
            use_gpu = torch.cuda.is_available(),
            weight_pl = 0.1,
            feat_dim = 96,
            temp = 1,
            num_classes = 10,
        )
    criterion.load_state_dict(checkpoint['criterion_state_dict']) 
    #speaker_fold = [speaker for speaker in os.listdir(train_data) if "unknown" not in speaker.lower()]
    #speaker_fold.append("unknown")
    #print(speaker_fold)
    #speaker_to_index = {speaker: idx for idx, speaker in enumerate(speaker_fold)}
    input, label = read_MFB(test_path) # input size:(n_frames, n_dims)
    input_processor = transforms.Compose([
        TruncatedInputfromMFB_NotRandom(),
        ToTensorInput()
    ])
    processed_inputs = input_processor(input)
    processed_inputs = processed_inputs.unsqueeze(1) 
    processed_inputs = processed_inputs.cuda()
    #print (processed_inputs.shape)
    embeddings, output = model(processed_inputs)
    output, _ = criterion(embeddings, output)
    predictions = output.data.max(1)[1]
    softmax = nn.Softmax(dim = 1)
    output = softmax(output)
    max_val, max_index = torch.max(output, 1)
    predicted_speaker = speaker_fold[max_index.item()]
    #return predicted_speaker, output
    if max_val.item() > threshold:
        return predicted_speaker, embeddings
    else:
        a = "unknown"
        return a, embeddings
def main():
    
    log_dir = 'TIMIT_UNKNOWN/TRAIN'         # Where the checkpoints are saved
    embedding_dir = 'TIMIT_UNKNOWN/TRAIN'    # Where embeddings are saved
    test_dir = 'TIMIT_UNKNOWN/TEST' # Where test features are saved
    
    use_cuda = True         # Use cuda or not
    embedding_size = 96    # Dimension of speaker embeddings
    cp_num = 99             # Epoch chọn
    n_classes = 10       # Số lượng người train
    test_frames = 200       # Chia nhỏ âm thanh test 
    # Tải model
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)

    # Get the dataframe for test DB
    #enroll_DB, test_DB = split_enroll_and_test(c.TEST_FEAT_DIR)
    
    # Tải d-vector
    #embeddings = load_enroll_embeddings(embedding_dir)
    parent_folder = Path(embedding_dir)
    spk_list = [f.name for f in parent_folder.iterdir() if f.is_dir()] # Lấy danh sách các folder con
    print (spk_list)  # Những người trong tập traintrain   
    # Set người test
    #spk_list = [name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))]
    #spk_list = list(embeddings.keys())

    '''
# Khởi tạo biến đếm dự đoán đúng và tổng số dự đoán cho từng người
    accuracy_per_speaker = {spk: {"correct": 0, "total": 0,"unknown":0} for spk in os.listdir(test_dir)}

# Duyệt qua từng thư mục người nói trong test_dir
    for test_speaker in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_speaker)

    # Lấy danh sách tất cả các file trong thư mục test của người nói
        all_files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

    # Duyệt qua từng file của người nói hiện tại
        for selected_file in all_files:
            selected_file_path = os.path.join(test_path, selected_file)

        # Thực hiện so sánh nhận dạng
            best_spk = perform_identification(use_cuda, model, embeddings, selected_file_path, test_frames, spk_list, test_speaker)

            # 
            accuracy_per_speaker[test_speaker]["total"] += 1

            # Evaluate if the prediction is correct or unknown or not at all
            if best_spk == test_speaker:
                accuracy_per_speaker[test_speaker]["correct"] += 1
            elif best_spk == "Unknown speaker" and test_speaker not in embeddings:
                accuracy_per_speaker[test_speaker]["unknown"] += 1

    # Caculate and print the accuracy for each speaker
    for spk, counts in accuracy_per_speaker.items():
        correct = counts["correct"]
        unknown = counts["unknown"]
        total = counts["total"]
        accuracy = (correct + unknown / total) * 100 if total > 0 else 0
        print(f"Accuracy for {spk}: {accuracy:.2f}%")

    # Overall Accuracy
    total_correct = sum(counts["correct"] for counts in accuracy_per_speaker.values()) + sum(counts["unknown"] for counts in accuracy_per_speaker.values())
    total_predictions = sum(counts["total"] for counts in accuracy_per_speaker.values())
    overall_accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    '''
   
    true_labels = []      # Nhãn thật (test_speaker)
    predicted_labels = [] # Kết quả dự đoán (best_spk)
    embedding_vectors = [] 

# Vòng lặp để thu thập dữ liệu từ frame của bạn
    for test_speaker in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_speaker)
        all_files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    
        for file in all_files:
            selected_file_path = os.path.join(test_path, file)

        # Gọi hàm để tạo vector nhúng + nhãn
            best_spk, embeddings = perform_classification(use_cuda, selected_file_path, model, test_frames)

            if embeddings is not None:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.detach().cpu().numpy()
                embedding_vectors.append(embeddings)  
                true_labels.append(test_speaker)
                predicted_labels.append(best_spk)

# Chuyển sang numpy array
    embedding_vectors = np.array(embedding_vectors)

# Chia nhỏ thành các batch để tránh OOM
    batch_size = 100
    num_batches = int(np.ceil(len(embedding_vectors) / batch_size))

# Tạo figure cho plot 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

# Danh sách các màu cho từng speaker
    unique_speakers = sorted(set(true_labels))
    color_map = plt.cm.get_cmap('tab10', len(unique_speakers))

# Vẽ theo từng batch
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(embedding_vectors))
    
        batch_embeddings = embedding_vectors[start:end]
        batch_labels = true_labels[start:end]
        batch_predictions = predicted_labels[start:end]
        print(batch_embeddings.shape)
        batch_embeddings = batch_embeddings.squeeze(1)  # Loại bỏ chiều 1
    # Giảm số chiều bằng t-SNE
        tsne = TSNE(n_components=3, perplexity=min(5, batch_embeddings.shape[0] - 1), n_iter = 2000, random_state=42)
        embeddings_3d = tsne.fit_transform(batch_embeddings)
    
    # Vẽ từng điểm trong batch
        for j in range(len(embeddings_3d)):
            x, y, z = embeddings_3d[j]
            if batch_labels[j] == "unknown":
                color = 'gray'  # Màu xám cho unknowns
                marker = 'x'    # Ký hiệu 'x' cho unknown
            else:
                speaker_idx = unique_speakers.index(batch_labels[j])
                color = color_map(speaker_idx)
                marker = 'o' if batch_labels[j] == batch_predictions[j] else '^' # '^' nếu dự đoán sai
        
            ax.scatter(x, y, z, c=[color], marker=marker, s=40, alpha=0.8)

    # Hiển thị tạm thời từng batch
        plt.pause(0.1)

# Tạo chú thích cho các người nóiss
    handles = []
    for i, speaker in enumerate(unique_speakers):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(i), markersize=8, label=speaker))
# Thêm chú thích cho "unknown"
    handles.append(plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=8, label='Unknown'))

# Thêm legend
    ax.legend(handles=handles, title="Speakers", loc="upper left", bbox_to_anchor=(1.05, 1))

# Đặt tiêu đề và hiển thị
    ax.set_title('Speaker Embedding Predictions with t-SNE')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')

    plt.show()
    
if __name__ == '__main__':
    main()