import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import configure as c
from pathlib import Path 
import torchvision.transforms as transforms
from DB_wav_reader import read_feats_structure
from redimnet.pretrained import ReDimNetWithClassifier 
from redimnet.model import ReDimNetWrap 
from SR_Dataset import read_MFB, TruncatedInputfromMFB, DvectorDataset, ToTensorInput, TruncatedInputfromMFB_NotRandom, collate_fn_feat_padded
from redimnet.hubconf import ReDimNet 
from SRPL import ARPLoss
from denoiser.denoiser import Denoiser   
from DB_wav_reader import read_feats_structure_train
import SRPL_evaluation as evaluation


def load_dataset(selected_speakers):
    full_DB = read_feats_structure_train(c.TEST_FEAT_DIR)

    # Đánh dấu known vs unknown
    full_DB['is_known'] = full_DB['speaker_id'].isin(selected_speakers)

    # Sắp xếp: known lên trước, unknown xuống dưới
    full_DB = full_DB.sort_values(by='is_known', ascending=False).reset_index(drop=True)

    # Chia lại test và test_out
    test_DB = full_DB[full_DB['is_known']].copy().reset_index(drop=True)
    test_out_DB = full_DB[~full_DB['is_known']].copy().reset_index(drop=True)

    file_loader = read_MFB
    transform = transforms.Compose([
        TruncatedInputfromMFB(),
        ToTensorInput()
    ])

    speaker_list = sorted(set(test_DB['speaker_id']))
    speaker_list_unknown = sorted(set(test_out_DB['speaker_id']))

    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    spk_to_idx_unknown = {spk: i for i, spk in enumerate(speaker_list_unknown)}

    test_dataset = DvectorDataset(DB=test_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    test_out_dataset = DvectorDataset(DB=test_out_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx_unknown)

    n_classes = len(speaker_list)
    print(f'\nTest set: {len(test_DB)} samples')
    print(f'Test unknown set: {len(test_out_DB)} samples')
    print(f'Total speakers: {n_classes}')
    return test_dataset, test_out_dataset, n_classes

# Tải mô hình đã lưu trữ vào để đánh giá

def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    backbone = ReDimNet('b0')
    #backbone = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=False)
    model = ReDimNetWithClassifier(backbone, num_classes = n_classes)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    checkpoint = torch.load('model_save_normalize/checkpoint_epoch_' + str(50) + '.pth')
    # Nạp tham số vào mô hình
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
def load_criterion():
    checkpoint = torch.load('model_save_normalize/checkpoint_epoch_' + str(50) + '.pth')
    criterion = ARPLoss(
            use_gpu = torch.cuda.is_available(),
            weight_pl = 0.1,
            feat_dim = 96,
            temp = 1,
            num_classes = 10,
        )
    criterion.load_state_dict(checkpoint['criterion_state_dict']) 
    return criterion
def load_denoiser(use_cuda):
    denoiser = Denoiser(feature_channels= 60, inference_steps=10)
    if use_cuda:
        denoiser.cuda()
    checkpoint = torch.load("denoiser_save/denoiser_epoch49.pt")
    denoiser.load_state_dict(checkpoint)    
    denoiser.eval()
    return denoiser

# Dự đoán kết quả mô hình   
def perform_classification(use_cuda, test_path, model, criterion, denoiser, test_frames, threshold = 0.947):
    train_data = c.TRAIN_FEAT_DIR
    speaker_fold = os.listdir(train_data)
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
    '''
    input_for_denoiser = processed_inputs.squeeze(0)#.squeeze(0)
    print(input_for_denoiser.size()) #(60,400)
    with torch.no_grad():
        denoised = denoiser(input_for_denoiser,input_for_denoiser, is_inference = True)
    denoised = denoised.unsqueeze(0)#.unsqueeze(0)
    '''
    embeddings, output, _ = model(processed_inputs)
    output, _ = criterion(embeddings)
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

# Tính toán độ chính xác khi duyệt qua tất cả các mẫu 

def calculate_accuracy(test_dir, use_cuda, model, criterion, denoiser, test_frames, parent_folder, threshold):
    spk_list = [f.name for f in parent_folder.iterdir() if f.is_dir()] # Lấy danh sách các folder con
    print (spk_list)
    total_predictions = 0
    correct_predictions = 0
    individual_accuracy = {spk: {"correct": 0, "total": 0} for spk in os.listdir(test_dir)}
    listed_total =0
    listed_correct = 0 
    unknown_total = 0
    unknown_correct = 0
    results = []
    for test_speaker in os.listdir(test_dir):
        test_path = os.path.join(test_dir, test_speaker)
        all_files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

        for file in all_files:
            total_predictions += 1
            individual_accuracy[test_speaker]["total"] += 1
            selected_file_path = os.path.join(test_path, file)

            # Thực hiện phân loại
            best_spk, embeddings = perform_classification(use_cuda, selected_file_path, model, criterion, denoiser, test_frames, threshold)

            # Xác định dự đoán đúng hay sai
            is_correct = False
            if test_speaker not in spk_list:
                unknown_total += 1
                if best_spk == "unknown":
                    correct_predictions += 1
                    individual_accuracy[test_speaker]["correct"] += 1
                    unknown_correct += 1
                    is_correct = True
            else:
                listed_total += 1
                if best_spk == test_speaker:
                    correct_predictions += 1
                    individual_accuracy[test_speaker]["correct"] += 1
                    listed_correct += 1
                    is_correct = True

            # In kết quả từng file
            status = "✅ ĐÚNG" if is_correct else "❌ SAI"
            print(f"[{status}] Speaker: {test_speaker:15} | File: {file:30} | Dự đoán: {best_spk}")
    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    known_accuracy = (listed_correct / listed_total) * 100 if listed_total > 0 else 0
    unknown_accuracy = (unknown_correct / unknown_total) * 100 if unknown_total > 0 else 0 
    return abs(known_accuracy - unknown_accuracy), known_accuracy, unknown_accuracy
    # Print overall accuracy

# Cân bằng độ chính xác 2 tác vụ

def find_optimal_threshold_binary_search(test_dir, use_cuda, model, criterion, denoiser, test_frames, parent_folder,
                                         tolerance=1.0, max_iters=20):
    low = 0.5
    high = 1.0
    best_threshold = 0.75
    for i in range(max_iters):
        threshold = (low + high) / 2
        diff, known_acc, unknown_acc = calculate_accuracy(
            test_dir, use_cuda, model, criterion, denoiser, test_frames, parent_folder, threshold
        )
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

def compute_metrics(model, criterion, test_loader, test_out_loader, use_cuda):
    n_correct, n_total = 0, 0
    # Chuyển model sang chế độ đánh giá
    model.eval()
    _pred_k, _pred_u, _labels = [], [], []
    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(test_loader):
            inputs, targets = data
            #current_sample = inputs.size(0)  # batch size
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            with torch.set_grad_enabled(False):
                x, y, _ = model(inputs)
                logits, _ = criterion(x)
                predictions = logits.data.max(1)[1]
                n_total += targets.size(0)
                n_correct += (predictions == targets.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(targets.data.cpu().numpy())
        for (data) in (test_out_loader):
            inputs, targets = data
    
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with torch.set_grad_enabled(False):
                x, y, _ = model(inputs)
                # x, y = net(data, return_feature=True)
                logits, loss = criterion(x)
                _pred_u.append(logits.data.cpu().numpy())
            # Tính toán độ chính xác
    acc = float(n_correct) * 100. / float(n_total)
    print('Acc: {:.5f}'.format(acc))
    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)
    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100
    return results['ACC'], results['OSCR']

def main():
    log_dir = c.TRAIN_FEAT_DIR         # Where the checkpoints are saved
    embedding_dir = c.TRAIN_FEAT_DIR     # Where embeddings are saved
    test_dir = c.TEST_FEAT_DIR # Where test features are saved
    use_cuda = True         # Use cuda or not
    embedding_size = 96    # Dimension of speaker embeddings
    cp_num = 99             # Epoch chọn      # Số lượng người train
    test_frames = 400       # Chia nhỏ âm thanh test 
    selected_speakers = ['Nguyen_Dinh_Minh', 'Nguyen_Huu_Trung', 'Nguyen_Thi_Cam_Ly', 'Pham_Quy_Long', 'Tran_Anh_Huy']
    test_dataset, test_out_dataset, n_classes = load_dataset(selected_speakers)
    # Tải model
    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    criterion = load_criterion()
    denoiser = load_denoiser(use_cuda)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=16,
                                                       shuffle=False,
                                                       collate_fn = collate_fn_feat_padded)
    test_out_loader = torch.utils.data.DataLoader(dataset = test_out_dataset, batch_size = 16, shuffle = False,
                                                    collate_fn = collate_fn_feat_padded)                     
    parent_folder = Path(embedding_dir)
    spk_list = [f.name for f in parent_folder.iterdir() if f.is_dir()] # Lấy danh sách các folder con
    print (spk_list)  # Những người trong tập train
    acc, oscr = compute_metrics(model, criterion, test_loader, test_out_loader, use_cuda)
    print(acc, oscr)
    find_optimal_threshold_binary_search(test_dir, use_cuda, model, criterion, denoiser, test_frames, parent_folder,
                                         tolerance=0.5, max_iters=20)
if __name__ == '__main__':
    main()