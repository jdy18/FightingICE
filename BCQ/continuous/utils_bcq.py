from typing import Tuple
# import d4rl
import gymnasium as gym
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math
import torch.nn.functional as FF
from dotmap import DotMap
from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd
import pathlib
import pickle

path='/Users/jin/Downloads/ICE/TD3+BC/Sample/Data_pretrain_0.pth'
BASE_CHECKPOINT_PATH = f'/Users/jin/Downloads/ICE/TD3+BC/actor_pt'

def load_buffer_ftg(expert_data_task: str) -> ReplayBuffer:
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

def save_checkpoint(actor):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    # checkpoint.env = ENV
    # checkpoint.iteration = iteration
    # checkpoint.stop_conditions = stop_conditions
    # checkpoint.hp = hp
    # CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    # if not rnn:
    CHECKPOINT_PATH = f'{BASE_CHECKPOINT_PATH}/bcq/'
    # else:
        # CHECKPOINT_PATH = f'{BASE_CHECKPOINT_PATH}/{encoder_name}/rnn/{experiment_id}/{iteration}/'

    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    # with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
    #     pickle.dump(checkpoint, f)
    # with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
    #     pickle.dump(type(actor), f)
    # with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
    #     pickle.dump(type(actor), f)
    torch.save(actor, CHECKPOINT_PATH + "actor.pt")
    # torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
    # torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer.pt")
    # torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer.pt")