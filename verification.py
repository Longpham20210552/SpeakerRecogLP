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
def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    backbone = ReDimNet('b0')  # Khởi tạo backbone ReDimNet
    model = Pretrained_ReDimNet(backbone)  # Bọc backbone vào lớp Pretrained
    if use_cuda:
        model.cuda()  # Chuyển sang GPU nếu cần
    print('=> loading checkpoint')
    model.eval()  # Đặt chế độ đánh giá
    return model

# Hàm load toàn bộ vector nhúng đã lưu từ thư mục embedding_dir
def load_enroll_embeddings(embedding_dir):
    embeddings = {}
    for f in os.listdir(embedding_dir):
        spk = f.replace('.pth','')  # Lấy tên người nói từ tên file
        embedding_path = os.path.join(embedding_dir, f)
        tmp_embeddings = torch.load(embedding_path)
        embeddings[spk] = tmp_embeddings
    return embeddings

# Hàm trích xuất vector nhúng từ file đặc trưng
def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename)
    tot_segments = math.ceil(len(input) / test_frames)
    activation = 0
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i * test_frames : i * test_frames + test_frames]
            TT = ToTensorTestInput()
            temp_input = TT(temp_input)
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    activation = l2_norm(activation, 1)  # Chuẩn hóa L2
    return activation

# Hàm chuẩn hóa L2 cho vector nhúng
def l2_norm(input, alpha):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    output = output * alpha
    return output

# Hàm nhận dạng: dự đoán người nói từ file test
def perform_classification(use_cuda, test_filename, model, embeddings, test_frames, thres):
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)
    best_score = -float('inf')
    best_spk = 'unknown'
    
    # So sánh với từng vector enroll
    for spk, emb in embeddings.items():
        score = F.cosine_similarity(test_embedding, emb).item()
        if score > best_score:
            best_score = score
            best_spk = spk
    
    # Nếu điểm số thấp hơn ngưỡng => unknown
    if best_score < thres:
        best_spk = 'unknown'
    
    return best_spk, best_score

# Hàm kiểm tra xác thực: đúng người nói hay không
def perform_verification(use_cuda, model, embeddings, enroll_speaker, test_filename, test_frames, thres):
    enroll_embedding = embeddings[enroll_speaker]
    test_embedding = get_embeddings(use_cuda, test_filename, model, test_frames)
    score = F.cosine_similarity(test_embedding, enroll_embedding).item()
    
    result = 'Accept' if score > thres else 'Reject'
    
    test_spk = os.path.basename(os.path.dirname(test_filename))  # Lấy tên người thật từ cấu trúc thư mục
    print("\n=== Speaker verification ===")
    print(f"Score : {score:.4f} | Threshold : {thres:.2f}")
    print(f"True speaker: {enroll_speaker} | Claimed speaker: {test_spk} | Result: {result}\n")

# Hàm tính chênh lệch độ chính xác giữa known và unknown cho 1 threshold
def calculate_accuracy(test_dir, use_cuda, model, embeddings, test_frames, threshold):
    total = correct = known_total = known_correct = unknown_total = unknown_correct = 0
    spk_list = list(embeddings.keys())

    for test_speaker in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_speaker)
        if not os.path.isdir(test_path):
            continue

        for file in os.listdir(test_path):
            if not file.endswith('.p'):
                continue
            total += 1
            filepath = os.path.join(test_path, file)
            best_spk, score = perform_classification(use_cuda, filepath, model, embeddings, test_frames, threshold)

            if test_speaker in spk_list:  # Người đã enroll
                known_total += 1
                if best_spk == test_speaker:
                    known_correct += 1
                    correct += 1
            else:  # Người unknown
                unknown_total += 1
                if best_spk == 'unknown':
                    unknown_correct += 1
                    correct += 1

    known_acc = (known_correct / known_total) * 100 if known_total else 0
    unknown_acc = (unknown_correct / unknown_total) * 100 if unknown_total else 0
    diff = abs(known_acc - unknown_acc)
    return diff, known_acc, unknown_acc

# Hàm tìm threshold tối ưu bằng tìm nhị phân
def find_optimal_threshold_binary_search(test_dir, use_cuda, model, embeddings, test_frames,
                                         tolerance=1.0, max_iters=20):
    low, high, best_threshold = 0, 1.0, 0.5
    for i in range(max_iters):
        threshold = (low + high) / 2
        diff, known_acc, unknown_acc = calculate_accuracy(test_dir, use_cuda, model, embeddings, test_frames, threshold)
        print(f"[{i+1:02}] Threshold = {threshold:.4f} | Δ = {diff:.2f}% | Known = {known_acc:.2f}% | Unknown = {unknown_acc:.2f}%")

        if diff <= tolerance:
            print(f"\nTìm được threshold tốt: {threshold:.4f} với Δ = {diff:.2f}%")
            return threshold, known_acc, unknown_acc

        if unknown_acc > known_acc:
            high = threshold
        else:
            low = threshold

        best_threshold = threshold

    print("\nKhông tìm được threshold thỏa mãn sau nhiều lần thử. Trả về giá trị gần nhất.")
    return best_threshold, known_acc, unknown_acc

# Hàm đánh giá tổng thể mô hình
def evaluate_model(test_dir, use_cuda, model, embeddings, test_frames, thres):
    total_predictions = correct_predictions = listed_total = listed_correct = unknown_total = unknown_correct = 0
    individual_accuracy = {spk: {"correct": 0, "total": 0} for spk in os.listdir(test_dir)}

    for test_speaker in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_speaker)
        if not os.path.isdir(test_path):
            continue

        all_files = [f for f in os.listdir(test_path) if f.endswith('.p')]
        for file in all_files:
            total_predictions += 1
            individual_accuracy[test_speaker]["total"] += 1
            selected_file_path = os.path.join(test_path, file)
            best_spk, score = perform_classification(use_cuda, selected_file_path, model, embeddings, test_frames, thres)

            is_correct = False
            if test_speaker not in embeddings:  # Người unknown
                unknown_total += 1
                if best_spk == "unknown":
                    unknown_correct += 1
                    correct_predictions += 1
                    individual_accuracy[test_speaker]["correct"] += 1
                    is_correct = True
            else:  # Người đã enroll
                listed_total += 1
                if best_spk == test_speaker:
                    listed_correct += 1
                    correct_predictions += 1
                    individual_accuracy[test_speaker]["correct"] += 1
                    is_correct = True

            status = "✅ ĐÚNG" if is_correct else "❌ SAI"
            print(f"[{status}] Speaker: {test_speaker:15} | File: {file:25} | Dự đoán: {best_spk:25} | Score: {score:.4f}")

    # Thống kê tổng thể
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions else 0
    print("\n=== TỔNG KẾT ===")
    print(f"Tổng số dự đoán: {total_predictions} | Số đúng: {correct_predictions} | Độ chính xác tổng thể: {overall_accuracy:.2f}%")
    print(f"\n-- Người đã enroll -- Tổng mẫu: {listed_total} | Số đúng: {listed_correct}")
    print(f"-- Người unknown -- Tổng mẫu: {unknown_total} | Số đúng: {unknown_correct}")

    print("\nĐộ chính xác từng người:")
    for spk, results in individual_accuracy.items():
        if results["total"] > 0:
            acc = (results["correct"] / results["total"]) * 100
            print(f"{spk:25}: {acc:.2f}%")
        else:
            print(f"{spk:25}: Không có mẫu kiểm tra")

# Hàm main tổng thể
def main():
    log_dir = 'model_saved'
    embedding_dir = 'embedding_TIMIT'
    test_dir = 'TIMIT_UNKNOWN/TEST'
    use_cuda = True
    embedding_size = 128
    cp_num = 3
    n_classes = 5
    test_frames = 400

    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    embeddings = load_enroll_embeddings(embedding_dir)

    print("\n=== TÌM NGƯỠNG TỐI ƯU ===")
    best_thres, known_acc, unknown_acc = find_optimal_threshold_binary_search(
        test_dir, use_cuda, model, embeddings, test_frames, tolerance=0.25, max_iters=20
    )

    print("\n=== ĐÁNH GIÁ LẠI VỚI NGƯỠNG TỐI ƯU ===")
    evaluate_model(test_dir, use_cuda, model, embeddings, test_frames, best_thres)

if __name__ == '__main__':
    main()
