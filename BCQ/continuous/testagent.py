import sys
sys.path.append('./')
import numpy as np
import os
import torch
from model import FeedForwardActor, RecurrentActor
from encoder import RawEncoder, FFTEncoder, MelSpecEncoder, SampleEncoder
from pyftg.ai_interface import AIInterface
from pyftg.struct import *
from pyftg.struct import CommandCenter

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


class TestAgent(AIInterface):
    def __init__(self, **kwargs):
        self.actor = kwargs.get('actor')
        self.device = kwargs.get('device')
        self.logger = kwargs.get('logger')
        self.n_frame = kwargs.get('n_frame')
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
        self.action_idx = None
        self.frameData = FrameData()
        self.cc = CommandCenter()


    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return False

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
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
        self.cc.skill_cancel()
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
            reward = self.get_reward()
            #self.collect_data_helper.put([obs, reward, False, None])

        # get action
        state = torch.tensor(obs, dtype=torch.float32)

        # if self.action_idx is None:
        #     self.action_idx = torch.zeros(len(self.actions))
        

        # if len(self.action_idx.shape) == 1:
        #     # # for i in range(0,14):
        #     # #     self.action_idx[i]=60
        #     self.action_idx[11]=1
        #     # self.action_idx[39]=1
        #     self.action_idx = self.action_idx.unsqueeze(0)

        # action_idx = self.action_idx
    
        if hasattr(self.actor, 'act'):  # 有act函数用act，否则用forward(原soundagent中的forward函数生成的是分布)
            self.action_idx = self.actor.act(state.to(self.device))
        else:
            self.action_idx = self.actor(state.reshape(-1).unsqueeze(0).to(self.device))
        action_idx = torch.argmax(self.action_idx) 
        self.cc.command_call(self.actions[action_idx])
        print(self.actions[action_idx])
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
