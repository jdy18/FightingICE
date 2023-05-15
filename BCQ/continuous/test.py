import sys
import argparse
import torch
from testagent import TestAgent
from model import RecurrentActor
from pyftg.gateway import Gateway
import logging
from encoder import SampleEncoder, RawEncoder, FFTEncoder, MelSpecEncoder
from tianshou.utils.net.common import MLP, Net
from tianshou.policy import BCQPolicy
from continuous import Critic,VAE, Perturbation
from gym import spaces #add

STATE_DIM = {
    1: {
        'conv1d': 160,
        'fft': 512,
        'mel': 2560
    },
    4: {
        'conv1d': 64,
        'fft': 512,
        'mel': 1280
    }
}

n_frame = 1
HIDDEN_SIZE = 512
RECURRENT_LAYERS = 1
ACTION_NUM = 40

#根据actor_name加载相应的网络， 需要测试其他网络时需要添加新的分支
def load_actor_model(encoder_name, actor_path, device, actor_name = 'RecurrentActor'):
    if actor_name == 'RecurrentActor':
        actor_model = RecurrentActor(STATE_DIM[n_frame][encoder_name], HIDDEN_SIZE, RECURRENT_LAYERS,
                                     get_sound_encoder(encoder_name, n_frame),
                                     action_num=ACTION_NUM)
        
    else:
        if actor_name == 'continuous_bcq':
            n_frame = 1
            observation_space=spaces.Box(low=-1.9, high=1.9, shape=(STATE_DIM[n_frame]['mel'],))
            action_space = spaces.Box(low=0, high=1, shape=(40,)) #add
            args.state_shape = observation_space.shape #env.observation_space.shape or env.observation_space.n
            args.action_shape = action_space.shape #env.action_space.shape or env.action_space.n
            args.state_dim = args.state_shape[0]
            args.action_dim = args.action_shape[0]
            args.max_action = action_space.high[0] #env.action_space.high[0]  # float
            # model
            encoder=get_sound_encoder('mel',n_frame)
            # perturbation network, add encoder into actor 
            net_a = MLP(
                input_dim=args.state_dim + args.action_dim,
                output_dim=args.action_dim,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            )
            actor = Perturbation(
                net_a, 
                max_action=args.max_action, 
                device=args.device, 
                phi=args.phi,
                encoder=encoder
            ).to(args.device)
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

            net_c1 = Net(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            net_c2 = Net(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device,
            )
            critic1 = Critic(net_c1, device=args.device,encoder=encoder).to(args.device)
            critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
            critic2 = Critic(net_c2, device=args.device,encoder=encoder).to(args.device)
            critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

            # vae
            # output_dim = 0, so the last Module in the encoder is ReLU
            vae_encoder = MLP(
                input_dim=args.state_dim + args.action_dim,
                hidden_sizes=args.vae_hidden_sizes,
                device=args.device,
            )
            if not args.latent_dim:
                args.latent_dim = args.action_dim * 2
            vae_decoder = MLP(
                input_dim=args.state_dim + args.latent_dim,
                output_dim=args.action_dim,
                hidden_sizes=args.vae_hidden_sizes,
                device=args.device,
            )
            vae = VAE(
                vae_encoder,
                vae_decoder,
                hidden_dim=args.vae_hidden_sizes[-1],
                latent_dim=args.latent_dim,
                max_action=args.max_action,
                device=args.device,
                pre_encoder=encoder
            ).to(args.device)
            vae_optim = torch.optim.Adam(vae.parameters())

            actor_model = BCQPolicy(
                actor,
                actor_optim,
                critic1,
                critic1_optim,
                critic2,
                critic2_optim,
                vae,
                vae_optim,
                device=args.device,
                gamma=args.gamma,
                tau=args.tau,
                lmbda=args.lmbda,
            )

            
    state_dict =actor_model.state_dict()
    model_state_dict = torch.load('/Users/jin/Downloads/FightingICE/log/BCQ-continuous-200epoch/epoch200policy.pth',map_location=torch.device('cpu'))
    actor_model.load_state_dict(model_state_dict,strict=False)
    # actor_model.load_state_dict(actor_state_dict)
    # sd  = actor_model.state_dict()
    # vae_dict = vae.state_dict()
    # actor_model.preprocess_net.load_state_dict

    actor_model.eval()
    actor_model.to(device)
    # actor_model.get_init_state(device)  # rnn模型需要初始化状态

    return actor_model


def get_sound_encoder(encoder_name, n_frame):
    encoder = None
    if encoder_name == 'conv1d':
        encoder = RawEncoder(frame_skip=n_frame)
    elif encoder_name == 'fft':
        encoder = FFTEncoder(frame_skip=n_frame)
    elif encoder_name == 'mel':
        encoder = MelSpecEncoder(frame_skip=n_frame)
    else:
        encoder = SampleEncoder()
    return encoder


def get_score(self_HP: list, opp_HP: list):
    total_rounds = len(self_HP)
    win_rounds = sum([self_HP[i] > opp_HP[i] for i in range(total_rounds)])
    win_ratio = win_rounds / total_rounds
    hp_diff = sum([self_HP[i] - opp_HP[i] for i in range(total_rounds)])
    hp_diff_avg = hp_diff / total_rounds
    return win_ratio, hp_diff_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, choices=['conv1d', 'fft', 'mel'], default='fft',
                        help='Choose an encoder for the Blind AI')
    parser.add_argument('--port', type=int, default=50051, help='Port used by DareFightingICE')
    parser.add_argument('--p2', choices=['Sandbox', 'MctsAi23i'], type=str, default='MctsAi23i', help='The opponent AI')
    parser.add_argument('--game_num', type=int, default=50, help='Number of games to play')
    parser.add_argument('--device', type=str, default='cpu', help='device for test')
    parser.add_argument('--actor_path', type=str, default='/Users/jin/Downloads/FightingICE/log/BCQ-continuous-200epoch/epoch200actor.pt', help='actor path')  # actor网络路径
    parser.add_argument('--actor_name', type=str, default='continuous_bcq', help='actor name')  # actor网络名字
    parser.add_argument('--save_path', type=str, default='/Users/jin/Downloads/FightingICE/BCQ/continuous/results/bcq_vs_MctsAi23i.txt', help='save path')  # 结果保存路径
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--phi", default=0.05)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-7)
    parser.add_argument("--vae-hidden-sizes", type=int, nargs="*", default=[512, 512])
    # default to 2 * action_dim
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    # Weighting for Clipped Double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)
    args = parser.parse_args()
    characters = ['ZEN']

    device = args.device
    actor_path = args.actor_path
    game_num = args.game_num
    encoder_name = args.encoder
    actor_name = args.actor_name
    save_path = args.save_path
    p2 = args.p2

    # logger config
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(' %(message)s')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #file_log 将log输出到文件
    file_handler = logging.FileHandler(filename=save_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Input parameters:')
    logger.info(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

    self_HP = []
    opp_HP = []

    for character in characters:
        # FFT GRU
        for _ in range(game_num):
            # actor模型加载，以blindAI的RNN为例
            actor_model = load_actor_model(encoder_name=encoder_name, actor_path=actor_path, device=device, actor_name=actor_name)
            agent = TestAgent(n_frame=n_frame, logger=logger, actor=actor_model, device=device)
            gateway = Gateway(port=50051)
            ai_name = 'FFTGRU'
            gateway.register_ai(ai_name, agent)
            print("Start game")
            gateway.run_game([character, character], [ai_name, p2], 1)
            print("After game")
            sys.stdout.flush()
            gateway.close()

            results = agent.results
            for round_result in results:
                self_HP.append(round_result.remaining_hps[0])
                opp_HP.append(round_result.remaining_hps[1])

    win_ratio, hp_diff_avg = get_score(self_HP, opp_HP)
    logger.info("\n win_ratio: %.3f, \n hp_diff_avg %.3f" %(hp_diff_avg, win_ratio))
