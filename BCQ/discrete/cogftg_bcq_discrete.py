#!/usr/bin/env python3

import argparse
import datetime
import os
import pickle
import pprint
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteBCQPolicy
from offline_trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from gymnasium import spaces #add
"""
make sure you have those files in the directory 
"""
from discrete import Actor
from Network import DQN# from examples.atari.atari_network import DQN
from fight_agent import get_sound_encoder,STATE_DIM
from utils_discrete import load_buffer_ftg #from utils_discrete import load_buffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    #TODO : what is target update freq
    parser.add_argument("--target-update-freq", type=int, default=8000)
    #TODO: change threshold to have generative network be either less behavioral clone/ Q-learning
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.3)
    #TODO: how imitation-logits-penalty works
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--update-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512])
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--resume-path", type=str, default='/Users/jin/Downloads/FightingICE-jin/jin-log/log/discrete-bcq-200epoch-0516/0policy.pth')
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--load-buffer-name", type=str, default="./expert_DQN_PongNoFrameskip-v4.hdf5"
    )
    parser.add_argument(
        "--buffer-from-rl-unplugged", action="store_true", default=False
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--first-dim", type=int, default=40
    )
    args = parser.parse_known_args()[0]
    return args


def test_discrete_bcq(args=get_args()):
    
    observation_space = spaces.Box(low=-1.9, high=1.9, shape=(800, 2))# args.state_shape = env.observation_space.shape or env.observation_space.n
    action_space = spaces.Box(low=0, high=1, shape=(40,))# args.action_shape = env.action_space.shape or env.action_space.n
    args.state_shape = observation_space.shape
    args.action_shape = action_space.shape
    # should be N_FRAMES x H x W
 
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder=get_sound_encoder('mel')
    # model
   
    feature_net = DQN(
        #TODO: change one to parameter 'channel'
       1, *args.state_shape, args.action_shape, device=args.device, features_only=True,batch_size=args.batch_size, first_dim=args.first_dim
    ).to(args.device)
    policy_net = Actor(
        preprocess_net = feature_net,
        action_shape = args.action_shape,
        device=args.device,
        hidden_sizes=args.hidden_sizes,
        softmax_output=False,
        encoder = encoder
    ).to(args.device)
    imitation_net = Actor(
        preprocess_net = feature_net,
        action_shape = args.action_shape,
        device=args.device,
        hidden_sizes=args.hidden_sizes,
        softmax_output=False,
        encoder = encoder
    ).to(args.device)
    actor_critic = ActorCritic(policy_net, imitation_net)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    # define policy
    policy = DiscreteBCQPolicy(
        policy_net, imitation_net, optim, args.gamma, args.n_step,
        args.target_update_freq, args.eps_test, args.unlikely_action_threshold,
        args.imitation_logits_penalty
    )
    # load a previous policy
    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location=args.device) 
        policy.load_state_dict(state_dict)
        print("Loaded agent from: ", args.resume_path)
        policy_net = policy.model
        torch.save(policy_net.state_dict(), os.path.join('/Users/jin/Downloads/FightingICE-jin/jin-log/log/discrete-bcq-200epoch-0516/', "actor.pth")) 

    # buffer
    buffer = load_buffer_ftg()# buffer = load_buffer(args.load_buffer_name)
    print("Replay buffer size:", len(buffer), flush=True)

    # collector
    test_collector = None   #Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "bcq"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy,num=0):
        torch.save(policy.state_dict(), os.path.join(log_path, str(num)+"policy.pth")) 
        torch.save(policy_net.state_dict(), os.path.join(log_path, str(num)+"actor.pth")) 

    def stop_fn(mean_rewards):
        return False

    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.update_per_epoch,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    pprint.pprint(result)
    # watch()


if __name__ == "__main__":
    test_discrete_bcq(get_args())
