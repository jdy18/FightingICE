from typing import Tuple

#import d4rl
import gymnasium as gym
#import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math
import torch.nn.functional as FF

from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd

path='../data/Sample_data.pth'


def normalize_all_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer
) -> Tuple[ReplayBuffer, RunningMeanStd]:
    # compute obs mean and var
    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    _eps = np.finfo(np.float32).eps.item()
    # normalize obs
    replay_buffer._meta["obs"] = (replay_buffer.obs -
                                  obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next -
                                       obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    return replay_buffer, obs_rms

def load_buffer_sequence(expert_data_task: str, action_one_hot:bool=True,sequence_len=120) -> ReplayBuffer:
    dataset = torch.load(path)
    if action_one_hot:
        dataset["actions"]=FF.one_hot(dataset["actions"], num_classes=40)

    dataset = dataset_process(dataset,sequence_len)

    replay_buffer = ReplayBuffer.from_data(
        obs=dataset["observations"],
        act=dataset["actions"],
        rew=dataset["rewards"],
        done=dataset["terminals"],
        obs_next=dataset["next_observations"],
        terminated=dataset["terminals"],
        truncated=np.zeros(len(dataset["terminals"]))
    )
    return replay_buffer

def dataset_process(dataset,sequence_len):
    dataset["observations"] = dataset["observations"]
    dataset["next_observations"] = dataset["next_observations"]
    dataset["terminals"] = dataset["terminals"].to(torch.bool)
    for key in dataset:
        dataset[key] = dataset[key].numpy()

    terminal_idx = np.where(dataset["terminals"]==1)[0] + 1  #每局游戏开始节点

    #拆分成长度为sequence_len的序列，结尾部分取【end-32,end】
    start_idx = []
    end_idx = []
    split_idx = 0
    for idx in terminal_idx:
        while split_idx + sequence_len < idx:
            start_idx.append(split_idx)
            end_idx.append(split_idx + sequence_len)
            split_idx += sequence_len
        start_idx.append(max(idx - sequence_len,0))
        end_idx.append(idx)
        split_idx = idx

    for key in dataset:
        dataset[key] = np.array([dataset[key][start:end] for start, end in zip(start_idx, end_idx)])
    #dataset["time"] = [np.arange(length) for length in (end_idx - start_idx)]

    return dataset

