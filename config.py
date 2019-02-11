import os
from os.path import join as pathjoin
from typing import NamedTuple, Tuple, Dict

import deepdish as dd
import numpy as np
import scipy.io as scio
import torch
from torch import nn

from mypath import *


# ============================ nn & cuda ==============================

# CUDA
CUDA_DEVICES = list(range(torch.cuda.device_count()))
OUT_CUDA_DEV = 1

# Files
F_HPARAMS = 'hparams.h5'


class HyperParameters(NamedTuple):
    # n_per_frame: int

    # method = 'mag'
    method = 'complex'
    # method = 'ifd'

    CHANNELS = dict(x='all', y='alpha',
                    fname_wav=False,
                    x_phase=True, y_phase=True,
                    )
    ch_base = 32
    dflt_kernel_size = (3, 3)
    dflt_pad = (1, 1)

    n_epochs = 169
    batch_size = len(CUDA_DEVICES) * 8
    learning_rate = 5e-4
    n_file = 20 * 500

    # p = 0.5  # Dropout p

    # lr scheduler
    StepLR = dict(step_size=5, gamma=0.8)

    CosineAnnealingLR = dict(T_max=10,
                             eta_min=0,
                             )

    CosineLRWithRestarts = dict(restart_period=10,
                                t_mult=2,
                                eta_threshold=1.5,
                                )

    weight_decay = 1e-5  # Adam weight_decay

    # weight_loss = (1, 0.7, 0.5)
    # weight_loss = (1, 0, 0)
    weight_loss = (1, 1, 1)

    # def for_MLP(self) -> Tuple:
    #     n_input = self.L_cut_x * self.n_per_frame
    #     n_hidden = 17 * self.n_per_frame
    #     n_output = self.n_per_frame
    #     return (n_input, n_hidden, n_output, self.p)

    def get_for_UNet(self) -> Tuple:
        ch_in = 4 if self.CHANNELS['x'] == 'all' else 1
        ch_out = 4 if self.CHANNELS['y'] == 'all' else 1
        if 'x_phase' in self.CHANNELS:
            ch_in += 1
        if 'y_phase' in self.CHANNELS:
            ch_out += 1

        return ch_in, ch_out, self.ch_base

    # noinspection PyProtectedMember
    def asdict(self):
        result = self._asdict()
        result['get_for_UNet'] = self.get_for_UNet()
        return result


hp = HyperParameters()

criterion = nn.MSELoss(reduction='sum')


# ========================= for audio utils ===========================

N_GRIFFIN_LIM = 20

DO_B_EQ = True

# metadata
_f_metadata = pathjoin(DICT_PATH['iv_train'], 'metadata.h5')
if os.path.isfile(_f_metadata):
    _metadata = dd.io.load(_f_metadata)
    # all_files = metadata['path_wavfiles']
    L_hop = int(_metadata['L_hop'])
    N_freq = int(_metadata['N_freq'])
    _N_LOC_TRAIN = int(_metadata['N_LOC'])
    Fs = int(_metadata['Fs'])
    N_fft: int = L_hop * 2

    # STFT/iSTFT arguments
    kwargs = dict(hop_length=L_hop, window='hann', center=True)
    KWARGS_STFT = dict(**kwargs, n_fft=N_fft, dtype=np.complex64)
    KWARGS_ISTFT = dict(**kwargs, dtype=np.float32)
    del kwargs

_f_metadata = pathjoin(DICT_PATH['iv_seen'], 'metadata.h5')
if os.path.isfile(_f_metadata):
    _metadata = dd.io.load(_f_metadata)
    _N_LOC_SEEN = int(_metadata['N_LOC'])

_f_metadata = pathjoin(DICT_PATH['iv_unseen'], 'metadata.h5')
if os.path.isfile(_f_metadata):
    _metadata = dd.io.load(_f_metadata)
    _N_LOC_UNSEEN = int(_metadata['N_LOC'])

N_LOC: Dict[str, int] = dict()
if '_N_LOC_TRAIN' in dir():
    # noinspection PyUnboundLocalVariable
    N_LOC['train'] = _N_LOC_TRAIN
if '_N_LOC_SEEN' in dir():
    # noinspection PyUnboundLocalVariable
    N_LOC['seen'] = _N_LOC_SEEN
if '_N_LOC_UNSEEN' in dir():
    # noinspection PyUnboundLocalVariable
    N_LOC['unseen'] = _N_LOC_UNSEEN

if len(N_LOC) < 3:
    print('Cannot get some of N_LOC')

# bEQspec
sft_dict = scio.loadmat(DICT_PATH['sft_data'],
                        variable_names=('bEQf',),
                        squeeze_me=True)
bEQf0: np.ndarray = sft_dict['bEQf'][:, 0][:, np.newaxis, np.newaxis]  # F, T, C
bEQf0_mag: np.ndarray = np.abs(bEQf0)
bEQf0_angle: np.ndarray = np.angle(bEQf0)
del sft_dict


# ========================== for IVDataset ============================

REFRESH_CONST = False

IV_DATA_NAME = dict(x='/IV_room', y='/IV_free',
                    x_phase='/phase_room', y_phase='/phase_free',
                    fname_wav='/fname_wav',
                    out='/IV_estimated',
                    out_phase='/phase_estimated',
                    )
CH_SLICES = {'all': dd.aslice[:, :, :],
             'alpha': dd.aslice[:, :, -1:],
             True: None,
             }


def is_ivfile(f: os.DirEntry) -> bool:
    return (f.name.endswith('.h5')
            and not f.name.startswith('metadata')
            and not f.name.startswith('normconst_'))


# ==================== for Normalization Classes ======================

EPS_FOR_LOG = 1e-10
USE_LOG = False
NORM_USING_ONLY_X = False


# ========================= dependent values ===========================

if hp.method == 'mag':
    N_LOSS_TERM = 1
    NORM_CLASS = 'LogMeanStdNormalization'
elif hp.method == 'complex':
    N_LOSS_TERM = 3
    if USE_LOG:
        NORM_CLASS = 'LogReImMeanStdNormalization'
    else:
        NORM_CLASS = 'ReImMeanStdNormalization'

CH_WITH_PHASE = dict(**hp.CHANNELS)

if 'x_phase' not in hp.CHANNELS:
    CH_WITH_PHASE['x_phase'] = True

if 'y_phase' not in hp.CHANNELS:
    CH_WITH_PHASE['y_phase'] = True

PERIOD_SAVE_STATE = hp.CosineLRWithRestarts['restart_period'] // 2
