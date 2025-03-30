import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms 
import time
import os
import numpy as np
import configure as c
import pandas as pd
from DB_wav_reader import read_feats_structure_train
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, ToTensorDevInput, DvectorDataset, collate_fn_feat_padded
import matplotlib.pyplot as plt
from redimnet.pretrained import ReDimNetWithClassifier
from redimnet.model import ReDimNetWrap
from redimnet.hubconf import ReDimNet 
from SRPL import ARPLoss
import SRPL_evaluation as evaluation
def load_dataset():
    # Tải trực tiếp từ file đã chia sẵn
    train_DB = read_feats_structure_train(c.TRAIN_FEAT_DIR)
    valid_DB = read_feats_structure_train(c.VALID_FEAT_DIR)
    
    file_loader = read_MFB # numpy array:(n_frames, n_dims)
    transform = transforms.Compose([
        TruncatedInputfromMFB(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(1, n_dims, n_frames)
    ])
    transform_T = ToTensorDevInput()
    
    speaker_list = sorted(set(train_DB['speaker_id'])) # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    
    train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx)
    
    n_classes = len(speaker_list) # Số người tham gia training
    
    print(f'\nTraining set: {len(train_DB)} samples')
    print(f'Validation set: {len(valid_DB)} samples')
    print(f'Total speakers: {n_classes}')
    
    return train_dataset, valid_dataset, n_classes
def load_out_dataset():
    """
    Tải out_dataset từ tập tin đặc trưng.
    Trả về: 
      - out_dataset: Dataset được trích xuất và tiền xử lý.
      - n_classes: Số lượng lớp (số người nói).
      - spk_to_idx: Ánh xạ từ speaker ID sang index.
    """
    # Load toàn bộ database
    out_DB = read_feats_structure_train(c.OUT_FEAT_DIR)  # OUT_FEAT_DIR là thư mục chứa các đặc trưng của out_dataset
    file_loader = read_MFB  # Hàm loader: numpy array (n_frames, n_dims)
    
    # Biến đổi (transform) cho dataset
    transform = transforms.Compose([
        TruncatedInputfromMFB(),  # numpy array: (1, n_frames, n_dims)
        ToTensorInput()  # torch tensor: (1, n_dims, n_frames)
    ])
    
    # Xác định danh sách người nói
    speaker_list = sorted(set(out_DB['speaker_id']))  # Lấy danh sách speaker_id duy nhất
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}  # Ánh xạ speaker_id sang index
    
    # Số lượng lớp (n_classes)
    n_classes = len(speaker_list)
    
    # Tạo dataset
    out_dataset = DvectorDataset(DB=out_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    
    return out_dataset, n_classes, spk_to_idx
'''
def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB = read_feats_structure_train(train_feat_dir)
    total_len = len(train_valid_DB) # 148642
    valid_len = int(total_len * valid_ratio/100.)
    train_len = total_len - valid_len
    shuffled_train_valid_DB = train_valid_DB.sample(frac=1).reset_index(drop=True)
    # Split the DB into train and valid set
    train_DB = shuffled_train_valid_DB.iloc[:train_len]
    valid_DB = shuffled_train_valid_DB.iloc[train_len:]
    # Reset the index
    train_DB = train_DB.reset_index(drop=True)
    valid_DB = valid_DB.reset_index(drop=True)
    print('\nTraining set %d utts (%0.1f%%)' %(train_len, (train_len/total_len)*100))
    print('Validation set %d utts (%0.1f%%)' %(valid_len, (valid_len/total_len)*100))
    print('Total %d utts' %(total_len))
    
    return train_DB, valid_DB
'''
def main():
    
    # Khởi tạo các hyperparameters
    use_cuda = True
    #torch.backends.cudnn.enabled = False
    
    embedding_size = 128    # Kích thước của D-vectors
    start = 1               # Vòng lặp bắt đầu 
    n_epochs = 100          # Training kéo dài 30 epoch
    end = start + n_epochs  
    lr = 5e-3        # Learning_rate = 0.12
    wd = 1e-4               # Điều khoản phạt hàm mất mát
    optimizer_type = 'adam'  # ex) sgd, adam, adagrad; chọn SGD làm phương pháp tối ưu
    batch_size = 16         # Kích thước batch training 
    valid_batch_size = 16   # Kích thước batch validation
    use_shuffle = True      # Có xáo trộn dữ liệu không?
    train_dataset, valid_dataset, n_classes = load_dataset()  # Tải tập dữ liệu
    out_dataset,_,_ = load_out_dataset()
    print('\nNumber of classes (speakers):\n{}\n'.format(n_classes))   # Số lượng người training: 233
    log_dir = 'model_saved_DATN_SRPL' # Lưu checkpoints sau mỗi epoch
    log_dir_2 = 'model_saved_05_lan3'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Khởi tạo mô hình và tham số ban đầu
    #model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    backbone = ReDimNet('b0')
    model = ReDimNetWithClassifier(backbone, num_classes = n_classes)
    """
    # Tải checkpoint
    checkpoint = torch.load(log_dir_2 + '/checkpoint_' + str(36) + '.pth')

    # Nạp tham số vào mô hình
    model.load_state_dict(checkpoint['state_dict'])
    """
    model.cuda()  # Sử dụng GPU
    
    # Định nghĩa hàm mất mát, hàm tối ưu, và hàm điều chỉnh học trong quá trình training
    criterion = ARPLoss(
        use_gpu = torch.cuda.is_available(),
        weight_pl = 0.1,
        feat_dim = 96,
        temp = 1 ,
        num_classes = 10
    )
    optimizer = create_optimizer(optimizer_type, model, lr, wd) # Sử dụng hàm tối ưu
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, min_lr=1e-10, verbose=1)  # Giảm lr sau 1 số epohất định nếu không có cải thiện về giá trị hàm mất mát
    
    # Tạo các data loader cho các tập dữ liệu train và valid, sử dụng xáo trộn dữ liệu
    
    # DataLoader cho train → dùng sampler cho cân bằng known/unknown
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=use_shuffle)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=valid_batch_size,
                                                       shuffle=False,
                                                       collate_fn = collate_fn_feat_padded)
    out_loader = torch.utils.data.DataLoader(dataset = out_dataset, batch_size = valid_batch_size, shuffle = False,
                                                    collate_fn = collate_fn_feat_padded)                     
    # Tạo mảng rỗng để theo dõi giá trị hàm mất mát training trung bình qua từng epoch
    avg_train_losses = []
    # Tạo mảng rỗng để theo dõi giá trị hàm mất mát validation trung bình qua từng epoch
    
    avg_valid_losses = []
    
    torch.cuda.empty_cache()
    results_history = {'train_loss': [], 'test_metrics': []}
    for epoch in range(start, end):

        train_loss, accuracy = train(model, criterion, optimizer, train_loader, epoch, use_cuda)
        results_history['train_loss'].append(train_loss)
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Training accuracy:{accuracy:.6f}")
    
        # Kiểm tra
        valid_loss, test_metrics = validate(model, criterion, valid_loader, out_loader, epoch, use_cuda)
        results_history['test_metrics'].append(test_metrics)
        print(f"Test Metrics: {test_metrics}")
        scheduler.step(valid_loss, epoch)
        # Lưu checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'train_loss': train_loss,
            'test_metrics': test_metrics,
        }
        torch.save(checkpoint, f"model_saved_SRPL_580/checkpoint_epoch_{epoch}.pth")
    
            # Hiển thị thông tin sau mỗi epoch
        print("-" * 40)
        print(f"Epoch {epoch} Complete")
        print(f"Accuracy: {test_metrics['ACC']:.2f}%")
        print(f"OSCR: {test_metrics['OSCR']:.2f}%")
        print("-" * 40)
                   
    np.save("results_history.npy", results_history)
    # visualize_the_losses(avg_train_losses, avg_valid_losses, fold_idx+1)
    
    visualize_metrics(results_history)
    

def train(model, criterion, optimizer, train_loader, epoch, use_cuda):
    batch_time = AverageMeter()
    losses = AverageMeter()
    #train_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    log_interval = 84
    # switch to train mode
    model.train()
    
    end = time.time()
    # pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data  # target size:(batch size,1), input size:(batch size, 1, dim, win)
        targets = targets.view(-1) # target size:(batch size)
        current_sample = inputs.size(0)  # batch size
       
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        x, y = model(inputs)
        logits, loss = criterion(x,y,targets)
        predictions = logits.data.max(1)[1]
        n_total += targets.size(0)
        n_correct += (predictions == targets.data).sum()
        losses.update(loss.item(), inputs.size(0)) # Tính trung bình hàm mất mát
        acc = float(n_correct) * 100. / float(n_total)
        # Tính toán gradient và cập nhật tham số
        optimizer.zero_grad() #Xóa gradient các tham số trước
        loss.backward()       #Backpropagation để tính toán tham số  
        optimizer.step()      #Cập nhật tham số mô hình

        # Thực thi các tác vụ đo thời gian
        batch_time.update(time.time() - end)  
        end = time.time()
    # Khi đã train hết dữ liệu, kết thúc 1 epoch, in ra màn hình
        if batch_idx % log_interval == 0:
            print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'.format(
                     epoch, batch_idx * len(inputs), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), 
                     batch_time=batch_time, loss=losses))
    return losses.avg, acc 
                     
def validate(model, criterion, val_loader, out_loader, epoch, use_cuda):
    
    # Khởi tạo các biến tính độ chính xác
    n_correct, n_total = 0, 0
    
    # Chuyển model sang chế độ đánh giá
    model.eval()
    _pred_k, _pred_u, _labels = [], [], []
    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            inputs, targets = data
            #current_sample = inputs.size(0)  # batch size
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = model(inputs)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                n_total += targets.size(0)
                n_correct += (predictions == targets.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(targets.data.cpu().numpy())
            
        for batch_idx, (data) in enumerate(out_loader):
            inputs, targets = data
        
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with torch.set_grad_enabled(False):
                x, y = model(inputs)
                # x, y = net(data, return_feature=True)
                logits, loss = criterion(x, y)
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
    results['OSCR'] = _oscr_socre * 100.

    return loss, results

class AverageMeter(object):
    # Lớp khởi tạo và lưu trữ các giá trị trung bình
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # Lưu trữ giá trị hiện tại, giá trị tổng, số lượng giá trị, giá trị trung bình
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(optimizer, model, new_lr, wd):
    # Tối ưu hàm mất mát
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,       # Sử dụng thuật toán Stochastic GD, với momentum = 0.9, độ
                              momentum=0.9, dampening=0,
                              weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=wd)
    return optimizer

def visualize_metrics(results_history):
    # Vẽ đồ thị hàm mất mát
    plt.figure(figsize=(12, 5))

    # Đồ thị hàm mất mát train
    plt.subplot(1, 2, 1)
    plt.plot(results_history['train_loss'], label='Train Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Đồ thị độ chính xác (ACC) và OSCR
    test_acc = [metrics['ACC'] for metrics in results_history['test_metrics']]
    test_oscr = [metrics['OSCR'] for metrics in results_history['test_metrics']]
    
    plt.subplot(1, 2, 2)
    plt.plot(test_acc, label='Accuracy (ACC)', color='green')
    plt.plot(test_oscr, label='OSCR', color='red')
    plt.title('Test Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()