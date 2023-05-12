import sys
sys.path.append('./')
import numpy as np
import os
import torch
from encoder import RawEncoder, FFTEncoder, MelSpecEncoder, SampleEncoder
# from pyftg.ai_interface import AIInterface
# from pyftg.struct import *

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

def get_sound_encoder(encoder_name, n_frame=1):
    encoder = None
    if encoder_name == 'conv1d':
        encoder = RawEncoder(frame_skip=n_frame)
    elif encoder_name == 'fft':
        encoder = FFTEncoder(frame_skip=n_frame)
    elif encoder_name == 'mel':
        encoder = MelSpecEncoder(frame_skip=n_frame)
    else:
        encoder = SampleEncoder(frame_skip=n_frame)
    return encoder
