import os
from enum import Enum
from typing import Any, Dict, NamedTuple, Tuple

import deepdish as dd
import numpy as np
import scipy.io as scio
import torch
# noinspection PyCompatibility
from dataclasses import dataclass, field
from numpy import ndarray
from torch import nn

from mypath import *


# noinspection PyArgumentList
class Channel(Enum):
    ALL = dict(sel=None)
    MAG = dict(sel=dd.aslice[:, :, -1:])
    NONE = None


# ============================ nn & cuda ==============================

CUDA_DEVICES = [*range(torch.cuda.device_count())]
# CUDA_DEVICES = [1, 2, 3]
# CUDA_DEVICES = [2, 3]
OUT_CUDA_DEV = 3


@dataclass
class HyperParameters:
    method: str = 'mag'
    # method: str = 'complex'
    # method: str = 'magphase'
    # method: str = 'magbpd'

    CHANNELS: Dict[str, Channel] = field(init=False)

    ch_base: int = 32
    dflt_kernel_size: Tuple[int, int] = (3, 3)
    dflt_pad: Tuple[int, int] = (1, 1)
    use_cbam: bool = False

    n_epochs: int = 210
    batch_size: int = len(CUDA_DEVICES) * 8
    learning_rate: float = 5e-4
    n_file: int = 20 * 500

    CosineLRWithRestarts: Dict[str, Any] = field(init=False)

    weight_decay: float = 1e-5  # Adam weight_decay

    weight_loss: tuple = (0.1, 1)  # complex
    # weight_loss: tuple = (0.3, 1, 0.08)  # magphase

    ch_in: int = field(init=False)
    ch_out: int = field(init=False)

    def __post_init__(self):
        self.CHANNELS = dict(fname_wav=Channel.NONE,
                             x=Channel.ALL,
                             # x=Channel.MAG,
                             y=Channel.MAG,
                             # x_phase=Channel.ALL,
                             # y_phase=Channel.ALL,
                             )
        self.CosineLRWithRestarts = dict(restart_period=10,
                                         t_mult=2,
                                         eta_threshold=1.5,
                                         )
        self.ch_in = 4 if self.CHANNELS['x'] == Channel.ALL else 1
        self.ch_out = 4 if self.CHANNELS['y'] == Channel.ALL else 1
        if 'x_phase' in self.CHANNELS:
            self.ch_in += 1
        if 'y_phase' in self.CHANNELS:
            self.ch_out += 1

    def get_for_UNet(self) -> Tuple[int, ...]:
        return self.ch_in, self.ch_out, self.ch_base

    # deprecated
    # n_per_frame: int

    # p = 0.5  # Dropout p

    # lr scheduler
    # StepLR: Dict[str, Any] = dict(step_size=5, gamma=0.8)
    #
    # CosineAnnealingLR: Dict[str, Any] = dict(
    #     T_max=10,
    #     eta_min=0,
    # )

    # def for_MLP(self) -> Tuple:
    #     n_input = self.L_cut_x * self.n_per_frame
    #     n_hidden = 17 * self.n_per_frame
    #     n_output = self.n_per_frame
    #     return (n_input, n_hidden, n_output, self.p)


hp = HyperParameters()

criterion = nn.MSELoss(reduction='sum')

# ==================== for Normalization Classes ======================

EPS_FOR_LOG = 1e-10
USE_LOG = True
NORM_USING_ONLY_X = False

# ========================= for audio utils ===========================

USE_GLIM_OUT_PHASE = False

N_GRIFFIN_LIM = 20

if DF == 'DirAC':
    DO_B_EQ = False
else:
    DO_B_EQ = True

# metadata
_f_metadata = DICT_PATH['dirspec_train'] / 'metadata.h5'
if _f_metadata.exists():
    _metadata = dd.io.load(_f_metadata)
    # all_files = metadata['path_wavfiles']
    L_hop = int(_metadata['L_hop'])
    N_freq = int(_metadata['N_freq'])
    _N_LOC_TRAIN = int(_metadata['N_LOC'])
    Fs = int(_metadata['Fs'])
    N_fft: int = (N_freq - 1) * 2

    # STFT/iSTFT arguments
    kwargs = dict(hop_length=L_hop, window='hann', center=True)
    KWARGS_STFT = dict(**kwargs, n_fft=N_fft, dtype=np.complex64)
    KWARGS_ISTFT = dict(**kwargs, dtype=np.float32)
    del kwargs

_f_metadata = DICT_PATH['dirspec_seen'] / 'metadata.h5'
if _f_metadata.exists():
    _metadata = dd.io.load(_f_metadata)
    _N_LOC_SEEN = int(_metadata['N_LOC'])
_f_metadata = DICT_PATH['dirspec_unseen'] / 'metadata.h5'
if _f_metadata.exists():
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
sft_dict = scio.loadmat(str(DICT_PATH['sft_data']),
                        variable_names=('bEQf',),
                        squeeze_me=True)
bEQf0: ndarray = sft_dict['bEQf'][:, 0][:, np.newaxis, np.newaxis]  # F, T(1), C(1)
bEQf0_mag: ndarray = np.abs(bEQf0)
bEQf0_angle: ndarray = np.angle(bEQf0)
del sft_dict

# ========================== for DirspecDataset ============================

REFRESH_CONST = False

SPEC_DATA_NAME = dict(x='/dirspec_room', y='/dirspec_free',
                      x_phase='/phase_room', y_phase='/phase_free',
                      x_bpd='/bpd_room', y_bpd='/bpd_free',
                      fname_wav='/fname_wav',
                      out='/dirspec_estimated',
                      out_phase='/phase_estimated',
                      out_bpd='/bpd_estimated',
                      )


def is_ivfile(f: os.DirEntry) -> bool:
    return (f.name.endswith('.h5')
            and not f.name.startswith('metadata')
            and not f.name.startswith('normconst_'))


# ========================= dependent values ===========================


KEY_TRANNORM = (hp.method, 'meanstd', USE_LOG)
if hp.method == 'mag' or hp.method == 'magbpd':
    N_LOSS_TERM = 1
    KEYS_TRANNORM = (KEY_TRANNORM,)
elif hp.method == 'complex':
    N_LOSS_TERM = 2
    KEYS_TRANNORM = (KEY_TRANNORM, ('mag', 'meanstd', True))
elif hp.method == 'magphase':
    N_LOSS_TERM = 3
    KEYS_TRANNORM = (('mag', 'meanstd', True), ('complex', 'meanstd', False))

CH_WITH_PHASE = dict(**hp.CHANNELS)

if 'x_phase' not in hp.CHANNELS:
    CH_WITH_PHASE['x_phase'] = Channel.ALL

if 'y_phase' not in hp.CHANNELS:
    CH_WITH_PHASE['y_phase'] = Channel.ALL

PERIOD_SAVE_STATE = hp.CosineLRWithRestarts['restart_period'] // 2

# ============================ files ==============================

F_HPARAMS = Path('hparams.h5')
S_F_RESULT = 'dirspec_{}'
F_LOG = Path('log.txt')
F_SCALARS = Path('scalars.json')
