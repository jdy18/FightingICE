'''
test.py
测试开始前，需要根据自己的actor网络修改load_actor_model（）函数来加载相应的模型
需要输入的参数：
actor_path: actor网络的保存位置，建议放在/model文件夹下
save_path:测试结果保存的位置，建议放在/results文件夹下
actor_name：atcor网络的名字，决定了如何加载模型
参考命令：
python test.py --actor_path ./model/actor.pt --actor_name TD3BC --save_path ./results/TD3BC_vs_MctsAi23i.txt
'''

import sys
import argparse
import torch
from td3bcagent import TD3BCAgent
from pyftg.gateway import Gateway
import logging

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
    parser.add_argument('--p2', choices=['Sandbox','Sounder' 'MctsAi23i'], type=str, default='MctsAi23i', help='The opponent AI')
    parser.add_argument('--game_num', type=int, default=50, help='Number of games to play')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='device for test')
    parser.add_argument('--actor_path', type=str, default='./model/actor.pth', help='actor path')  # actor网络路径
    parser.add_argument('--actor_name', type=str, default='TD3BC', help='actor name')  # actor网络名字
    parser.add_argument('--save_path', type=str, default='./results/ppopretrain_vs_MctsAi23i.txt', help='save path')  # 结果保存路径

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
            agent = TD3BCAgent(n_frame=n_frame, logger=logger, actor_path=actor_path,  actor_name=actor_name, device=device)
            gateway = Gateway(port=50051)
            ai_name = 'FFTGRU'
            gateway.register_ai(ai_name, agent)
            print("Start game")
            while True:
                try:
                    gateway.run_game([character, character], [ai_name, p2], 1)
                    break
                except:
                    pass

            print("After game")
            sys.stdout.flush()
            gateway.close()

            results = agent.results
            for round_result in results:
                self_HP.append(round_result.remaining_hps[0])
                opp_HP.append(round_result.remaining_hps[1])

    win_ratio, hp_diff_avg = get_score(self_HP, opp_HP)
    logger.info("\n win_ratio: %.3f, \n hp_diff_avg %.3f" %(hp_diff_avg, win_ratio))
