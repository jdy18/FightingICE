'''
test.py
测试开始前，需要根据自己的actor网络修改load_actor_model（）函数来加载相应的模型
需要输入的参数：
ports: 启动的端口位置，默认为[50051,50052,50053]，即同时开启三个，最终运行的round数为len(ports) * game_num * 3
actor_path: actor网络的保存位置，建议放在/model文件夹下
save_path:测试结果保存的位置，建议放在/results文件夹下
actor_name：atcor网络的名字，决定了如何加载模型
game_path: 游戏本体所在目录，默认为'../Game/'
script_name:游戏脚本的名字，默认windows环境，即为‘'run-windows-amd64.bat’

游戏环境设置：
1.将游戏本体放在game_path下
2.修改启动脚本，如windows脚本在前面增加以下代码
if "%1"=="" (
  set PORT=50051
) else (
  set PORT=%1
)
同时在启动游戏的命令的末尾加上如下参数
  --port %PORT%

linux环境下据说是这么写，还没测过，不行的话就把game_thread那段代码注释掉手动启动游戏：
先加上
if [ -z "$1" ]; then
  PORT="50051"
else
  PORT="$1"
fi
同时在启动游戏的命令的末尾加上如下参数
  --port $PORT
'''

import sys
import argparse
import torch

from testagent import TestAgent
import time
from model import RecurrentActor
from pyftg.gateway import Gateway
import logging
from discrete import Actor
from encoder import SampleEncoder, RawEncoder, FFTEncoder, MelSpecEncoder,get_sound_encoder
from tianshou.policy import DiscreteCRRPolicy
from tianshou.utils.net.common import Net
from tqdm import tqdm
import os
import subprocess
import threading
import time


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
        actor_state_dict = torch.load(actor_path)
        actor_model.load_state_dict(actor_state_dict)
        actor_model.to(device)
        actor_model.get_init_state(device)  # rnn模型需要初始化状态

    if actor_name == 'CRR':
        feature_net = Net(
            STATE_DIM[n_frame][encoder_name],
            hidden_sizes=[512],
            device=device,
        )
        encoder = get_sound_encoder('mel', n_frame=n_frame)
        actor_model = Actor(
            feature_net,
            ACTION_NUM,
            hidden_sizes=[512],
            device=device,
            softmax_output=True,
            encoder=encoder
        ).to(args.device)

        actor_dict = torch.load(actor_path)
        actor_model.load_state_dict(actor_dict, strict=False)

    return actor_model


#获得声音encoder
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

# 执行启动游戏脚本的函数
def run_game(port, stop_event, game_path, script_name):
    # command = "java -cp ..\Game\FightingICE.jar;./lib/*;./lib/lwjgl/*;./lib/lwjgl/natives/windows/amd64/*;./lib/grpc/*; " \
    #           "Main --limithp 400 400 --grpc-auto --non-delay 0 --port %d --blind-player 2" % (port)
    script_name = "run-windows-amd64.bat"
    command = [script_name, str(port)]
    process = subprocess.Popen(command, shell=True, cwd=game_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    #在适当的时候退出游戏,还没调通，先手动关闭吧
    while(1):
        time.sleep(10)
        if stop_event.is_set():
            process.terminate()
            return


#根据结果计算评价指标
def get_score(results):
    score = {}
    self_HP = [result.remaining_hps[0] for result in results]
    opp_HP = [result.remaining_hps[1] for result in results]
    elapsed_frame = [result.elapsed_frame for result in results]
    total_rounds = len(self_HP)
    score['round_num'] = total_rounds

    win_rounds = sum([self_HP[i] > opp_HP[i] for i in range(total_rounds)])
    score['win_ratio'] = win_rounds / total_rounds
    hp_diff = sum([self_HP[i] - opp_HP[i] for i in range(total_rounds)])
    score['hp_diff_avg'] = hp_diff / total_rounds

    remain_time = 0
    for i in range(total_rounds):
        if self_HP[i] > opp_HP[i]:
            remain_time += (3600 - elapsed_frame[i])/60
    score['Speed'] = remain_time / total_rounds / 60
    score['RemainHP'] = sum(self_HP) / total_rounds / 400
    score['Advantage'] = 0.5 * (sum(self_HP) - sum(opp_HP)) / total_rounds/ 400
    score['Damage'] = 1 - sum(opp_HP) / total_rounds / 400

    return score

#初始化logger
def log_init(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(' %(message)s')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #file_log 将log输出到文件
    file_handler = logging.FileHandler(filename=args.save_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, choices=['conv1d', 'fft', 'mel'], default='mel',
                        help='Choose an encoder for the Blind AI')
    parser.add_argument('--ports', type=list, default=[50051,50052], help='Port used by DareFightingICE')
    parser.add_argument('--p2', choices=['Sounder', 'MctsAi23i'], type=str, default='MctsAi23i', help='The opponent AI')
    parser.add_argument('--game_num', type=int, default=2, help='Number of games to play')
    parser.add_argument('--device', type=str, default='cpu', help='device for test')
    parser.add_argument('--game_path', type=str, default='../Game/', help='game path')  # 游戏本体路径
    parser.add_argument('--script_name', type=str, default='run-windows-amd64.bat', help='name of game script')  # 游戏启动的脚本名，默认windows
    parser.add_argument('--actor_path', type=str, default='log/crr_ramdomdata_lr1e-7/230525-202917/epoch200policy.pth', help='actor path')  # actor网络路径
    parser.add_argument('--actor_name', type=str, default='CRR', help='actor name')  # actor网络名字
    parser.add_argument('--save_path', type=str, default='./results/crr_prelr_1e-5.txt', help='save path')  # 结果保存路径

    args = parser.parse_args()
    characters = ['ZEN']

    device = args.device
    actor_path = args.actor_path
    game_num = args.game_num
    encoder_name = args.encoder
    actor_name = args.actor_name
    save_path = args.save_path
    p2 = args.p2
    game_path = args.game_path
    script_name = args.script_name
    ports = args.ports

    # logger config
    logger = log_init(args)
    logger.info('Input parameters:')
    logger.info(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

    # 创建线程并启动
    game_threads = []
    test_threads = []
    results = []
    stop_event = threading.Event()

    #运行游戏
    for port in ports:
        game_thread = threading.Thread(target=run_game, args=(port, stop_event, game_path, script_name))
        game_thread.start()
        game_threads.append(game_thread)

    #在某个端口上测试的函数
    def test(port, game_num):
        characters = ['ZEN']
        for character in characters:
            # FFT GRU
            for _ in tqdm(range(game_num), desc="port:%d"%port):
                actor_model = load_actor_model(encoder_name=encoder_name, actor_path=actor_path, device=device,
                                               actor_name=actor_name)
                agent = TestAgent(n_frame=n_frame, logger=logger, actor=actor_model, device=device)
                #启动游戏
                gateway = Gateway(port=port)
                ai_name = 'ai'
                gateway.register_ai(ai_name, agent)
                print("Start game")
                gateway.run_game([character, character], [ai_name, p2], 1)
                print("After game")
                sys.stdout.flush()
                gateway.close()

                for round_result in agent.results:
                    results.append(round_result)
        return

    #遍历所有端口，同时进行测试
    for port in ports:
        test_thread = threading.Thread(target=test, args=(port, game_num))
        test_thread.start()
        test_threads.append(test_thread)

    # 等待所有线程完成
    for thread in test_threads:
        thread.join()
    stop_event.set()

    #获得评价指标
    score = get_score(results)
    for key in score:
        logger.info(key + ": " + str(score[key]))





