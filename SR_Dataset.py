import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import pickle  # Đọc file pickle chứa đặc trưng
import numpy as np
import configure as c
from DB_wav_reader import read_DB_structure

# Đọc đặc trưng MFB và nhãn từ file .p
def read_MFB(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)

    feature = feat_and_label['feat']  # Đặc trưng Mel Filter Bank, kích thước (n_frames, dim)
    label = feat_and_label['label']

    # Lấy toàn bộ tín hiệu, giữ nguyên số frame
    start_sec, end_sec = 0, 0
    start_frame = int(start_sec / 0.01)
    end_frame = len(feature) - int(end_sec / 0.01)

    ori_feat = feature
    feature = feature[start_frame:end_frame, :]

    # Kiểm tra độ dài tín hiệu tối thiểu
    assert len(feature) > 20, (
        f'length is too short. len:{len(feature)}, ori_len:{len(ori_feat)}, file:{filename}'
    )

    return feature, label

# Cắt đoạn chính giữa tín hiệu, không ngẫu nhiên
class TruncatedInputfromMFB_NotRandom(object):
    def __init__(self, input_per_file=1):
        super().__init__()
        self.input_per_file = input_per_file  # Số đoạn muốn lấy từ 1 file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)
        win_size = c.NUM_WIN_SIZE
        half_win_size = int(win_size / 2)

        # Lặp tín hiệu nếu quá ngắn
        while num_frames - half_win_size <= half_win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)

        center_index = num_frames // 2  # Tính chỉ số giữa
        for _ in range(self.input_per_file):
            start_index = max(0, center_index - half_win_size)
            end_index = start_index + win_size
            frames_slice = frames_features[start_index:end_index]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)

# Cắt đoạn ngẫu nhiên từ tín hiệu
class TruncatedInputfromMFB(object):
    def __init__(self, input_per_file=1):
        super().__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)
        win_size = c.NUM_WIN_SIZE
        half_win_size = int(win_size / 2)

        # Lặp tín hiệu nếu quá ngắn
        while num_frames - half_win_size <= half_win_size:
            frames_features = np.append(frames_features, frames_features[:num_frames, :], axis=0)
            num_frames = len(frames_features)

        for _ in range(self.input_per_file):
            j = random.randrange(half_win_size, num_frames - half_win_size)
            frames_slice = frames_features[j - half_win_size:j + half_win_size]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)

# Cắt sliding window trên tín hiệu theo từng frame (dùng cho test)
class TruncatedInputfromMFB_test(object):
    def __init__(self, input_per_file=1):
        super().__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)

        for _ in range(self.input_per_file):
            for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                network_inputs.append(frames_slice)

        return np.array(network_inputs)

# Cắt sliding window cho tín hiệu, định dạng phù hợp cho CNN
class TruncatedInputfromMFB_CNN_test(object):
    def __init__(self, input_per_file=1):
        super().__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)

        for _ in range(self.input_per_file):
            for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                network_inputs.append(frames_slice)

        network_inputs = np.expand_dims(network_inputs, axis=1)  # Thêm chiều kênh

        assert network_inputs.ndim == 4, f'Data is not a 4D tensor. size:{np.shape(network_inputs)}'
        return np.array(network_inputs)

# Chuyển numpy array sang tensor 3D (batch, dim, n_win)
class ToTensorInput(object):
    def __call__(self, np_feature):
        if isinstance(np_feature, np.ndarray):
            return torch.from_numpy(np_feature.transpose((0, 2, 1))).float()

# Chuyển numpy array sang tensor 3D (1, dim, n_win), dùng cho dev/test
class ToTensorDevInput(object):
    def __call__(self, np_feature):
        np_feature = np.expand_dims(np_feature, axis=0)
        assert np_feature.ndim == 3, f'Data is not a 3D tensor. size:{np.shape(np_feature)}'
        return torch.from_numpy(np_feature.transpose((0, 2, 1))).float()

# Chuyển numpy array sang tensor 4D (1, 1, dim, n_win), dùng cho test
class ToTensorTestInput(object):
    def __call__(self, np_feature):
        np_feature = np.expand_dims(np_feature, axis=0)
        np_feature = np.expand_dims(np_feature, axis=1)
        assert np_feature.ndim == 4, f'Data is not a 4D tensor. size:{np.shape(np_feature)}'
        return torch.from_numpy(np_feature.transpose((0, 1, 3, 2))).float()

# Hàm gom batch, thực hiện padding theo độ dài lớn nhất trong batch
def collate_fn_feat_padded(batch):
    batch.sort(key=lambda x: x[0].shape[2], reverse=True)
    feats, labels = zip(*batch)

    labels = torch.stack(labels, 0).view(-1)
    lengths = [feat.shape[2] for feat in feats]
    max_length = lengths[0]

    padded_features = torch.zeros(len(feats), feats[0].shape[0], feats[0].shape[1], max_length).float()

    for i, feat in enumerate(feats):
        num_frames = feat.shape[2]
        while max_length > num_frames:
            feat = torch.cat((feat, feat[:, :, :num_frames]), 2)
            num_frames = feat.shape[2]

        padded_features[i, :, :, :] = feat[:, :, :max_length]

    return padded_features, labels

# Dataset huấn luyện dvector
class DvectorDataset(data.Dataset):
    def __init__(self, DB, loader, spk_to_idx, transform=None):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx

    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        feature, label = self.loader(feat_path)
        label = self.spk_to_idx[label]
        label = torch.Tensor([label]).long()

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def __len__(self):
        return self.len

# Dataset mở rộng, gán nhãn unknown nếu nhãn không nằm trong tập
class DvectorDataset_2(data.Dataset):
    def __init__(self, DB, loader, spk_to_idx, transform=None):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx

        if 'unknown' not in self.spk_to_idx:
            self.spk_to_idx['unknown'] = len(self.spk_to_idx)

    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        feature, label = self.loader(feat_path)

        if label not in self.spk_to_idx:
            label = 'unknown'

        label = self.spk_to_idx[label]
        label = torch.Tensor([label]).long()

        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def __len__(self):
        return self.len

# Hàm main kiểm tra dataset
def main():
    train_DB = read_DB_structure(c.TRAIN_DATAROOT_DIR)

    transform = transforms.Compose([
        TruncatedInputfromMFB_CNN_test(),
        ToTensorDevInput()
    ])

    file_loader = read_MFB

    speaker_list = sorted(set(train_DB['speaker_id']))
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}

    batch_size = 128

    Dvector_train_dataset = DvectorDataset(
        DB=train_DB,
        loader=file_loader,
        transform=transform,
        spk_to_idx=spk_to_idx
    )

    Dvector_train_loader = torch.utils.data.DataLoader(
        dataset=Dvector_train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

if __name__ == '__main__':
    main()
