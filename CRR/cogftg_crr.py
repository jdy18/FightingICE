#!/usr/bin/env python3

import argparse
import datetime
import os
import pickle
import pprint

import numpy as np
import torch
from gym import spaces
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteCRRPolicy
# from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from utils import load_buffer_ftg
from encoder import STATE_DIM, MelSpecEncoder,get_sound_encoder
from discrete import Actor, Critic
from offline_trainer import offline_trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pretraindata")
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--policy-improvement-mode", type=str, default="exp")
    parser.add_argument("--ratio-upper-bound", type=float, default=20.)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--min-q-weight", type=float, default=10.)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--update-per-epoch", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512])
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--expert-data-task", type=str, default="halfcheetah-expert-v2"
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
    args = parser.parse_known_args()[0]
    return args


def test_discrete_crr(args=get_args()):
    n_frame = 1
    observation_space=spaces.Box(low=-1.9, high=1.9, shape=(STATE_DIM[n_frame]['mel'],))
    action_space = spaces.Box(low=0, high=1, shape=(40,))
    args.state_shape = observation_space.shape #env.observation_space.shape or env.observation_space.n
    args.action_shape = action_space.shape #env.action_space.shape or env.action_space.n
    args.max_action = action_space.high[0] #env.action_space.high[0]  # float

    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(action_space.low), np.max(action_space.high))

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    print("Max_action", args.max_action)


    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    encoder=get_sound_encoder('mel')

    feature_net = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )

    actor = Actor(
        feature_net,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax_output=False,
        encoder=encoder
    ).to(args.device)
    critic = Critic(
        feature_net,
        hidden_sizes=args.hidden_sizes,
        last_size=np.prod(args.action_shape),
        device=args.device,
        encoder=encoder
    ).to(args.device)

    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    # define policy
    policy = DiscreteCRRPolicy(
        actor,
        critic,
        optim,
        args.gamma,
        policy_improvement_mode=args.policy_improvement_mode,
        ratio_upper_bound=args.ratio_upper_bound,
        beta=args.beta,
        min_q_weight=args.min_q_weight,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # buffer
    buffer = load_buffer_ftg(args.expert_data_task, action_one_hot=False)
    print("Replay buffer size:", len(buffer), flush=True)


    # collector
    test_collector = None

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "crr"

    log_name = os.path.join(str(args.algo_name) + "_" + str(args.task), now)
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
        

    def save_best_fn(policy,num= 0):
        torch.save(policy.state_dict(), os.path.join(log_path, str(num)+"policy.pth")) #.state_dict()

    def stop_fn(mean_rewards):
        return False

    # # watch agent's performance
    # def watch():
    #     print("Setup test envs ...")
    #     policy.eval()
    #     test_envs.seed(args.seed)
    #     print("Testing agent ...")
    #     test_collector.reset()
    #     results = test_collector.collect(n_episode=args.test_num, render=args.render)
    #     pprint.pprint(results)
    #     rew = results["rews"].mean()
    #     print(f'Mean reward (over {results["n/ep"]} episodes): {rew}')

    # if args.watch:
    #     watch()
    #     exit(0)

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

    # pprint.pprint(results)
    # watch()


if __name__ == "__main__":
    test_discrete_crr(get_args())
