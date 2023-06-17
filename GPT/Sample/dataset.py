import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math

class FightingICEDataset(Dataset):
    def __init__(self, path, sequence_len=32, ROLL_OUT=3600):
        self.trajectories = torch.load(path)
        self.sequence_len = sequence_len
        truncated_seq_len = torch.clamp(self.trajectories["seq_len"] - sequence_len + 1, 0, ROLL_OUT)  # 舍弃轨迹最后sequence_len个状态，因为以这些状态作为开始的序列无法凑齐长度为sequence_len的序列
        self.cumsum_seq_len = np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))

    def __len__(self):
        return self.cumsum_seq_len[-1]

    def __getitem__(self, start_idx):
        eps_idx = np.digitize(start_idx, bins=self.cumsum_seq_len, right=False) - 1  # 当前idx在哪个轨迹中
        seq_idx = start_idx - self.cumsum_seq_len[eps_idx]  # 当前idx在该轨迹中的索引
        series_idx = np.linspace(seq_idx, seq_idx + self.sequence_len - 1, num=self.sequence_len, dtype=np.int64)  # 以此idx为首的序列
        return self.trajectories['actions'][eps_idx, series_idx], self.trajectories['action_probabilities'][eps_idx, series_idx], self.trajectories['states'][eps_idx, series_idx], self.trajectories['rewards'][eps_idx, series_idx]

if __name__ == '__main__':
    datasets = FightingICEDataset('./Data_pretrain.pth')
    dataloader = DataLoader(datasets, batch_size=64, shuffle=True)
    for data in dataloader:
        while True:
            a=1
            pass
        print(data)

