import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import random 
import time
import os
import numpy as np
import configure as c
import pandas as pd
from DB_wav_reader import read_feats_structure_train
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, ToTensorDevInput, DvectorDataset, collate_fn_feat_padded
from model.model import background_resnet
import matplotlib.pyplot as plt
from redimnet.pretrained import ReDimNetWithClassifier
#from model.Pretrained_ReDimNet import model 
from redimnet.model import ReDimNetWrap
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler, ConcatDataset
from redimnet.hubconf import ReDimNet


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
def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB = read_feats_structure_train(train_feat_dir)
    
    # Nhóm theo từng speaker
    grouped = train_valid_DB.groupby('speaker_id')
    
    train_list = []
    valid_list = []
    
    for speaker, group in grouped:
        n_total = len(group) // 3 # mỗi người có n file gốc + 2n file augment = 3n file
        n_train = int(n_total * (1 - valid_ratio)) # 75% file gốc cho train
        n_valid = n_total - n_train # 25% file gốc cho validation
        
        # Lấy n_train file gốc cho train và các file augment tương ứng
        train_files = group.iloc[:n_train]
        train_augments = group.iloc[n_total:n_total + n_train].append(group.iloc[2 * n_total:2 * n_total + n_train])
        
        # Lấy n_valid file gốc cho valid và các file augment tương ứng
        valid_files = group.iloc[n_train:n_total]
        valid_augments = group.iloc[n_total + n_train:n_total + n_total].append(group.iloc[2 * n_total + n_train:2 * n_total + n_total])
        
        # Nối các file vào tập train và valid
        train_list.append(train_files)
        train_list.append(train_augments)
        valid_list.append(valid_files)
        valid_list.append(valid_augments)
    
    # Kết hợp lại thành dataframe
    train_DB = pd.concat(train_list).reset_index(drop=True)
    valid_DB = pd.concat(valid_list).reset_index(drop=True)

    print('\nTraining set %d utts (%0.1f%%)' %(len(train_DB), (len(train_DB)/len(train_valid_DB)*100)))
    print('Validation set %d utts (%0.1f%%)' %(len(valid_DB), (len(valid_DB)/len(train_valid_DB)*100)))
    print('Total %d utts' %(len(train_valid_DB)))
    
    return train_DB, valid_DB
    
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


class DvectorDataset(Dataset):
    def __init__(self, DB, loader, spk_to_idx, transform=None):
        self.DB = DB
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx
    
    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        feature, label = self.loader(feat_path)
        label = self.spk_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            feature = self.transform(feature)
        return feature, label
    
    def __len__(self):
        return len(self.DB)

def split_train_dev(train_feat_dir):
    train_valid_DB = read_feats_structure_train(train_feat_dir)

    train_DB = []
    valid_DB = []
    
    speaker_list = train_valid_DB['speaker_id'].unique()
    
    for spk in speaker_list:
        spk_files = train_valid_DB[train_valid_DB['speaker_id'] == spk]
        original_files = spk_files[~spk_files['filename'].str.contains('noise')]
        noise_files = spk_files[spk_files['filename'].str.contains('noise')]
        
        original_files = original_files.sort_values(by='filename').reset_index(drop=True)
        noise_files = noise_files.sort_values(by='filename').reset_index(drop=True)
        
        if spk == 'unknown':
            train_original = original_files.iloc[:18]
            train_noise = noise_files[noise_files['filename'].str.contains('|'.join(train_original['filename'].str.replace('.p', '')))]

            valid_original = original_files.iloc[18:24]
            valid_noise = noise_files[noise_files['filename'].str.contains('|'.join(valid_original['filename'].str.replace('.p', '')))]
        else:
            train_original = original_files.iloc[:6]
            train_noise = noise_files[noise_files['filename'].str.contains('|'.join(train_original['filename'].str.replace('.p', '')))]

            valid_original = original_files.iloc[6:8]
            valid_noise = noise_files[noise_files['filename'].str.contains('|'.join(valid_original['filename'].str.replace('.p', '')))]

            train_DB.append(train_original)
            train_DB.append(train_noise)
            valid_DB.append(valid_original)
            valid_DB.append(valid_noise)

        if spk == 'unknown':
            train_DB.append(train_original)
            train_DB.append(train_noise)
            valid_DB.append(valid_original)
            valid_DB.append(valid_noise)

    train_DB = pd.concat(train_DB).reset_index(drop=True)
    valid_DB = pd.concat(valid_DB).reset_index(drop=True)
    
    print(f"\nTraining set: {len(train_DB)} files")
    print(f"Validation set: {len(valid_DB)} files")
    print(f"Total: {len(train_DB) + len(valid_DB)} files")

    return train_DB, valid_DB

def load_dataset():
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR)
    
    file_loader = read_MFB
    transform = transforms.Compose([
        TruncatedInputfromMFB(),
        ToTensorInput()
    ])
    transform_T = ToTensorDevInput()
    
    speaker_list = sorted(set(train_DB['speaker_id']))
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    
    # ✅ Tạo tập train và valid
    train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx)
    
    # ✅ Sampler cho cân bằng giữa các lớp known/unknown
    known_sampler = RandomSampler(train_dataset, replacement=True, num_samples=18)
    unknown_sampler = RandomSampler(train_dataset, replacement=True, num_samples=18)

    n_classes = len(speaker_list)

    return train_dataset, valid_dataset, known_sampler, unknown_sampler, n_classes
'''
def main():
    
    # Khởi tạo các hyperparameters
    use_cuda = True
    #torch.backends.cudnn.enabled = False
    
    embedding_size = 128    # Kích thước của D-vectors
    start = 1               # Vòng lặp bắt đầu 
    n_epochs = 200          # Training kéo dài 30 epoch
    end = start + n_epochs  
    lr = 2e-3              # Learning_rate = 0.12
    wd = 1e-4               # Điều khoản phạt hàm mất mát
    optimizer_type = 'adam'  # ex) sgd, adam, adagrad; chọn SGD làm phương pháp tối ưu
    batch_size = 16         # Kích thước batch training 
    valid_batch_size = 16   # Kích thước batch validation
    use_shuffle = True      # Có xáo trộn dữ liệu không?
    train_dataset, valid_dataset, n_classes = load_dataset()  # Tải tập dữ liệu
    print('\nNumber of classes (speakers):\n{}\n'.format(n_classes))   # Số lượng người training: 233
    log_dir = 'model_saved_DATN' # Lưu checkpoints sau mỗi epoch
    log_dir_2 = 'model_saved_05_lan3'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Khởi tạo mô hình và tham số ban đầu
    #model = background_resnet(embedding_size=embedding_size, num_classes=n_classes)
    #backbone = ReDimNetWrap()
    #backbone = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=False)
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
    criterion = nn.CrossEntropyLoss()   # Hàm mất mát Cross Entropy Loss
    optimizer = create_optimizer(optimizer_type, model, lr, wd) # Sử dụng hàm tối ưu
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-15, verbose=1)  # Giảm lr sau 1 số epohất định nếu không có cải thiện về giá trị hàm mất mát
    
    # Tạo các data loader cho các tập dữ liệu train và valid, sử dụng xáo trộn dữ liệu
    
    # DataLoader cho train → dùng sampler cho cân bằng known/unknown
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=use_shuffle)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=valid_batch_size,
                                                       shuffle=False,
                                                       collate_fn = collate_fn_feat_padded)
                         
    # Tạo mảng rỗng để theo dõi giá trị hàm mất mát training trung bình qua từng epoch
    avg_train_losses = []
    # Tạo mảng rỗng để theo dõi giá trị hàm mất mát validation trung bình qua từng epoch
    
    avg_valid_losses = []
    
    torch.cuda.empty_cache()

    for epoch in range(start, end):

        # Training 1 epoch 
        train_loss = train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes)
        
        # Kiểm tra sau khi train xong
        valid_loss = validate(valid_loader, model, criterion, use_cuda, epoch)
        # Hiệu chỉnh lr sau mỗi epoch
        scheduler.step(valid_loss, epoch)
        
        # Tính toán hàm mất mát lưu vào các mảng
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        # Thực hiện lưu model
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   '{}/checkpoint_{}.pth'.format(log_dir, epoch))
    
    # Tìm epoch có hàm mất mát đánh giá (valid) nhỏ nhất, khuyến khích sử dụng
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    print('Lowest validation loss at epoch %d' %minposs)
    
    # Vẽ đồ thị hàm loss (không sử dụng)
    visualize_the_losses(avg_train_losses, avg_valid_losses)
    

def train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    
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
        _, output = model(inputs) # out size:(batch size = 64 , #classes ), for softmax
        
        # Tính toán độ chính xác training
        n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc_temp = 100. * n_correct / n_total
        train_acc.update(train_acc_temp, inputs.size(0)) # Tính độ chính xác trung bình
         
        loss = criterion(output, targets) # Tính toán hàm mất mát
        losses.update(loss.item(), inputs.size(0)) # Tính trung bình hàm mất mát
        
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
                    'Loss {loss.avg:.4f}\t'
                    'Acc {train_acc.avg:.4f}'.format(
                     epoch, batch_idx * len(inputs), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), 
                     batch_time=batch_time, loss=losses, train_acc=train_acc))
    return losses.avg
                     
def validate(val_loader, model, criterion, use_cuda, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()
    
    # Khởi tạo các biến tính độ chính xác
    n_correct, n_total = 0, 0
    
    # Chuyển model sang chế độ đánh giá
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            inputs, targets = data
            current_sample = inputs.size(0)  # batch size
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # Tính toán đầu ra mô hình
            _, output = model(inputs)
            
            # Tính toán độ chính xác
            n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
            n_total += current_sample
            val_acc_temp = 100. * n_correct / n_total

            # Update độ chính xác mới, tính độ chính xác trung bìnhbình
            val_acc.update(val_acc_temp, inputs.size(0))
            
            # Tính toán giá trị hàm mất mát và update trung bình
            loss = criterion(output, targets)
            losses.update(loss.item(), inputs.size(0))

            # Tính toán thời gian kiểm tra
            batch_time.update(time.time() - end)
            end = time.time()
        
            # In ra các giá trị hàm mất mát trung bình, độ chính xác trung bình
        print('  * Validation: '
                  'Loss {loss.avg:.4f}\t'
                  'Acc {val_acc.avg:.4f}'.format(
                  loss=losses, val_acc=val_acc))
    
    return losses.avg

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

def visualize_the_losses(train_loss, valid_loss):
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')
    
    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 3.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig('loss_plot_2_5e-4_16_10class.png', bbox_inches='tight')

if __name__ == '__main__':
    main()