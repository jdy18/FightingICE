import sys
sys.path.append('./')
import numpy as np
import os
import torch
from pyftg.ai_interface import AIInterface
from pyftg.struct import *
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

def set_logger():
    # logger config
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(' %(message)s')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file_log 将log输出到文件
    file_handler = logging.FileHandler(filename='./ppopretrain_vs_MctsAi23i.txt', mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

    #logger.info('Input parameters:')
    #logger.info(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

class CASAgent(AIInterface):
    def __init__(self,n_frame=1, logger=set_logger(), actor_path='./model/actor.pth',actor_name='CAS',device="cuda" if torch.cuda.is_available() else "cpu",**kwargs):
        actor_model = self.load_actor_model( actor_path=actor_path,  actor_name=actor_name, device=device)
        self.actor = actor_model
        self.device = device
        self.logger = logger
        self.n_frame = n_frame
        self.trajectories_data = None
        self.actions = "AIR_A", "AIR_B", "AIR_D_DB_BA", "AIR_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_DA", "AIR_DB", \
                   "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "BACK_JUMP", "BACK_STEP", \
                   "CROUCH_A", "CROUCH_B", "CROUCH_FA", "CROUCH_FB", "CROUCH_GUARD", "DASH", "FOR_JUMP", "FORWARD_WALK", \
                   "JUMP", "NEUTRAL", "STAND_A", "STAND_B", "STAND_D_DB_BA", "STAND_D_DB_BB", "STAND_D_DF_FA", \
                   "STAND_D_DF_FB", "STAND_D_DF_FC", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_FA", "STAND_FB", \
                   "STAND_GUARD", "THROW_A", "THROW_B"
        self.audio_data = None
        self.raw_audio_memory = None
        self.just_inited = True
        self.pre_framedata: FrameData = None
        self.nonDelay: FrameData = None
        self.round_count = 0
        self.results = []
        self.count = 3

        #self.initialize()

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return False

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
        self.frameData = FrameData()
        self.cc = CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.isGameJustStarted = True
        return 0

    def close(self):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool, non_delay: FrameData):
        # Load the frame data every time getInformation gets called
        self.frameData = frame_data
        self.cc.set_frame_data(self.frameData, self.player)
        # nonDelay = self.frameData
        self.pre_framedata = self.nonDelay if self.nonDelay is not None else non_delay
        self.nonDelay = non_delay
        self.isControl = is_control
        self.currentFrameNum = self.frameData.current_frame_number  # first frame is 14

    def round_end(self, round_result: RoundResult):
        self.results.append(round_result)
        self.logger.info("p1_HP:{}".format(round_result.remaining_hps[0]))
        self.logger.info("p2_HP:{}".format(round_result.remaining_hps[1]))
        self.logger.info("elapsed_frame:{}".format(round_result.elapsed_frame))
        self.just_inited = True
        self.raw_audio_memory = None
        self.round_count += 1
        self.logger.info('Finished {} round'.format(self.round_count))


    def input(self):
        return self.inputKey

    @torch.no_grad()
    def processing(self):
        if self.frameData.empty_flag or self.frameData.current_frame_number <= 0:
            self.isGameJustStarted = True
            return

        self.inputKey.empty()
        # self.cc.skill_cancel()
        obs = self.raw_audio_memory
        if self.just_inited:
            self.just_inited = False
            if obs is None:
                obs = np.zeros((800 * self.n_frame, 2))
            #self.collect_data_helper.put([obs])
            terminal = 1
        elif obs is None:
            obs = np.zeros((800 * self.n_frame, 2))
            #self.collect_data_helper.put([obs])
        else:
            terminal = 0
            # reward = self.get_reward()
            #self.collect_data_helper.put([obs, reward, False, None])

        # get action
        state = torch.tensor(obs, dtype=torch.float32)

        if hasattr(self.actor, 'act'):  # 有act函数用act，否则用forward(原soundagent中的forward函数生成的是分布)
            action_idx = self.actor.act(state.unsqueeze(0).to(self.device)).float()
        else:
            action = self.actor(state.flatten().unsqueeze(0).to(self.device))[0].float()
        action_idx = torch.argmax(action)
        # if action_idx==5:
        # action[0,5]=0
        # action[0, 26] = 0
        # action[0, 30] = 0
        action_idx=torch.argmax(action)
        # action_idx=8
        self.count += 1
        if self.count > 3:
            if np.random.rand() < 0.05:
                self.count = 0
                self.cc.skill_cancel()
                self.cc.command_call('STAND_D_DF_FC')

            else:
                self.cc.skill_cancel()
                self.cc.command_call(self.actions[action_idx])

        self.inputKey = self.cc.get_skill_key()


        # action_dist = self.actor(state.unsqueeze(0).to(self.device), terminal=torch.tensor(terminal).float())
        # action = action_dist.sample()
        # self.cc.command_call(self.actions[action])
        # self.inputKey = self.cc.get_skill_key()

        # put to helper
        #self.collect_data_helper.put_action(action)
        # if self.rnn:
        #     self.collect_data_helper.put_actor_hidden_data(self.actor.hidden_cell.squeeze(0).to(self.device))
        #



    def get_reward(self):
        offence_reward = self.pre_framedata.get_character(not self.player).hp - self.nonDelay.get_character(
            not self.player).hp
        defence_reward = self.nonDelay.get_character(self.player).hp - self.pre_framedata.get_character(self.player).hp
        return offence_reward + defence_reward

    def set_last_hp(self):
        self.last_my_hp = self.nonDelay.get_character(self.player).hp
        self.last_opp_hp = self.nonDelay.get_character(not self.player).hp

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data
        # process audio
        try:
            byte_data = self.audio_data.raw_data_as_bytes
            np_array = np.frombuffer(byte_data, dtype=np.float32)
            raw_audio = np_array.reshape((2, 1024))
            raw_audio = raw_audio.T
            raw_audio = raw_audio[:800, :]
        except Exception as ex:
            raw_audio = np.zeros((800, 2))
        if self.raw_audio_memory is None:
            # self.logger.info('raw_audio_memory none {}'.format(raw_audio.shape))
            self.raw_audio_memory = raw_audio
        else:
            self.raw_audio_memory = np.vstack((raw_audio, self.raw_audio_memory))
            # self.raw_audio_memory = self.raw_audio_memory[:4, :, :]
            self.raw_audio_memory = self.raw_audio_memory[:800 * self.n_frame, :]

        # append so that audio memory has the first shape of n_frame
        increase = (800 * self.n_frame - self.raw_audio_memory.shape[0]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.vstack((np.zeros((800, 2)), self.raw_audio_memory))

    # 根据actor_name加载相应的网络， 需要测试其他网络时需要添加新的分支
    def load_actor_model(self, actor_path, device, actor_name='RecurrentActor'):

        actor_model = torch.load(actor_path, map_location=torch.device(device)).actor
        actor_model.device = torch.device(device)
        return actor_model
