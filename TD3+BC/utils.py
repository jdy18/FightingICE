from typing import Tuple

import d4rl
import gymnasium as gym
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math
import torch.nn.functional as FF

from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd

# path='./Sample/Data_random_1.pth'
def load_buffer_ftg(expert_data_task: str,path='./Sample/Data_random_1.pth') -> ReplayBuffer:
    dataset = torch.load(path)
    # dataset = d4rl.qlearning_dataset(gym.make(expert_data_task))
    dataset["actions"]=FF.one_hot(dataset["actions"], num_classes=40)
    replay_buffer = ReplayBuffer.from_data(
        obs=dataset["observations"].view(dataset["observations"].shape[0],-1).numpy(),
        act=dataset["actions"].numpy(),
        rew=dataset["rewards"].numpy(),
        done=dataset["terminals"].to(torch.bool).numpy(),
        obs_next=dataset["next_observations"].view(dataset["observations"].shape[0],-1).numpy(),
        terminated=dataset["terminals"].to(torch.bool).numpy(),
        truncated=np.zeros(len(dataset["terminals"]))
    )
    return replay_buffer


def load_buffer(buffer_path: str) -> ReplayBuffer:
    with h5py.File(buffer_path, "r") as dataset:
        buffer = ReplayBuffer.from_data(
            obs=dataset["observations"],
            act=dataset["actions"],
            rew=dataset["rewards"],
            done=dataset["terminals"],
            obs_next=dataset["next_observations"],
            terminated=dataset["terminals"],
            truncated=np.zeros(len(dataset["terminals"]))
        )
    return buffer


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
