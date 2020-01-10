import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np
# noinspection PyCompatibility
from dataclasses import asdict, dataclass, field
from numpy import ndarray
import scipy.io as scio


# noinspection PyArgumentList
class Channel(Enum):
    ALL = slice(None)
    LAST = slice(-1, None)
    NONE = None

# The instance of this class is fully initialized after its `parse_argument` method is called.
@dataclass
class _HyperParameters:
    # -- devices --
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)  # GPU idx or 'cpu'
    out_device: Union[int, str] = 2  # The output of the DNN is gathered into this device.
    num_workers: int = 4  # No. of DataLoader workers

    # -- select dataset --
    feature: str = 'SIV'  # SIV (also used for Single model)
    # feature: str = 'DV'  # DV
    # feature: str = 'mulspec'  # 32ch mag + 32ch phase
    # feature: str = 'mulspec4'  # 4ch mag + 4ch phase
    room_train: str = 'room1+2+3'
    room_test: str = 'room1+2+3'
    room_create: str = ''

    model_name: str = 'UNet'

    # -- dirspec parameters --
    fs: int = 16000
    n_fft: int = 512
    l_frame: int = 512  # The case `n_fft != l_frame` hasn't been validated.
    n_freq: int = 257
    l_hop: int = 256
    n_data_per_room: int = 23 * 300  # 20 train RIRs + 3 valid RIRs
    n_test_per_room: int = 10 * 100  # 10 test RIRs

    # idx of mics in eigenmike. This starts from 0.
    # In the eigenmike spec sheet, the idx starts from 1.
    chs_mulspec4: Tuple[int] = (5, 6, 20, 21)

    # -- pre-processing --
    eps_for_log: float = 1e-5  # this is from librosa.amplitude_to_db
    use_log: bool = True

    # This should be True once after the dataset has been changed.
    # If there isn't "normconst.npz" file in the training set directory
    # or `refresh_const == True`,
    # normalization constant is calculated before starting training or testing.
    # If not, the existing "normconst.npz" file is used for normalization.
    refresh_const: bool = False

    # -- training --
    n_data: int = 0  # 0 for using all data
    train_ratio: float = 0.77  # $20/23 \approx 0.77$

    # maximum no. of epochs.
    # Training can be stopped earlier if early stopping criterion exists.
    n_epochs: int = 60

    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4  # Adam or AdamW weight_decay
    threshold_stop: float = 0.99  # for early stopping criterion. this is not used by default.

    # -- reconstruction --
    n_gla_iter: int = 20  # 0 for not using Griffin-Lim
    momentum_gla: float = 0.  # momentum for fast Griffin-Lim. 0 for vanilla GL.
    use_das_phase: bool = True  # If True, the phase of DAS beamforming output is used.
    das_err: str = ''
    # das_err: str = '_err5'
    # das_err: str = '_err10'

    # -- summary --
    period_save_state: int = 4  # period to save DNN model (unit: epoch)
    eval_with_y_ph: bool = False  # If True, evaluation is proceeded with true (anechoic) phase.
    draw_test_fig: bool = False  # If True, the spectrogram of test data is drawn in tensorboard.
    add_test_audio: bool = True  # If True, time-domain audio of test data is added to tensorboard.

    # No. of test data to save hidden block outputs.
    n_save_block_outs: int = 0

    # -- paths --
    # logdir will be converted to type Path in the init_dependent_vars function
    logdir: str = f'./result/test'
    path_speech: Path = Path('./data/TIMIT')  # directory of original speech
    path_feature: Path = Path('./data')  # directory of SIV, DV, mulspec data
    # path_feature: Path = Path('./backup/dirspecs')
    sfx_featuredir: str = ''
    # suffix_datadir: str = '_13_10'

    # -- file names --
    form_feature: str = '{:05d}_{:04d}_{}_{:02d}.npz'  # idx, i_speech, room, i_loc
    form_result: str = 'dirspec_{}.mat'

    # -- initialized in __post_init__ --
    channels: Dict[str, Channel] = field(init=False)
    UNet: Dict[str, Any] = field(init=False)
    scheduler: Dict[str, Any] = field(init=False)
    spec_data_names: Dict[str, str] = field(init=False)

    # -- dependent variables --
    dummy_input_size: Tuple = None
    dict_path: Dict[str, Path] = None
    kwargs_stft: Dict[str, Any] = None
    kwargs_istft: Dict[str, Any] = None
    channels_w_ph: Dict[str, Channel] = None
    folder_das_phase: str = ''

    def __post_init__(self):
        # DirspecDataset object load the data that are needed only.
        # for all: path_speech=NONE, y=LAST
        # SIV model: x=ALL, x_mag=NONE
        # DV model: x=ALL, x_mag=NONE
        # Single model: x=LAST, x_mag=NONE
        # Mulspec32, Mulspec4 model: x=ALL, x_mag=ALL
        self.channels = dict(path_speech=Channel.NONE,
                             x=Channel.ALL,
                             x_mag=Channel.NONE,
                             y=Channel.LAST,
                             # x_phase=Channel.ALL,
                             # y_phase=Channel.ALL,
                             )

        # ch_base: The no. of filter of the first block.
        #          2*ch_base for the second block, 4*ch_base for the third...
        # depth: no. of blocks in encoder path.
        self.UNet = dict(ch_base=64,
                         depth=4,
                         )

        # Read "adamwr/README.md"
        self.scheduler = dict(restart_period=4,
                              t_mult=2,
                              eta_threshold=1.5,
                              )

        # In DNN codes, data names are 'x', 'y', etc.
        # In "create.py" and the feature file, data names are 'dirspec_room', etc.
        self.spec_data_names = dict(x='dirspec_room', y='dirspec_free',
                                    x_mag='mag_room',
                                    x_phase='phase_room', y_phase='phase_free',
                                    path_speech='path_speech',
                                    out='dirspec_estimated',
                                    )

    def init_dependent_vars(self):
        self.logdir = Path(self.logdir)
        # nn
        if self.channels['x'] == Channel.ALL:
            if self.feature == 'mulspec':
                ch_in = 64  # 32 mag & 32 phase
            elif self.feature == 'mulspec4':
                ch_in = 8  # 4 mag & 4 phase
            else:
                ch_in = 4  # SIV, DV model
        else:
            ch_in = 1  # Single model

        self.UNet['ch_in'] = ch_in
        self.UNet['ch_out'] = 1

        self.dummy_input_size = (  # for torchsummary
            ch_in,
            self.n_freq,
            int(2**np.floor(np.log2(4 / 3 * self.n_freq))),  # the closest $2^n$
        )

        # path
        if self.room_create:
            self.room_train = self.room_create
            self.room_test = self.room_create
        form = f'{self.feature}_{{}}{self.sfx_featuredir}'
        p_f_train = self.path_feature / f'{form.format(self.room_train)}/TRAIN'
        p_f_test = self.path_feature / f'{form.format(self.room_test)}/TEST'
        self.dict_path = dict(
            sft_data=self.path_feature / 'sft_data_32ms.mat',
            RIR_Ys=self.path_feature / f'RIR_Ys_{self.room_create}.mat',

            speech_train=self.path_speech / 'TRAIN',
            speech_test=self.path_speech / 'TEST',

            feature_train=p_f_train,
            feature_seen=p_f_test / 'SEEN',
            feature_unseen=p_f_test / 'UNSEEN',

            normconst_train=p_f_train / 'normconst.npz',

            figures=Path('./figures'),
        )

        # STFT/iSTFT parameters
        self.kwargs_stft = dict(hop_length=self.l_hop, n_fft=self.n_fft)
        self.kwargs_istft = dict(hop_length=self.l_hop)

        # For reconstruction, phase data should be loaded.
        self.channels_w_ph = dict(**self.channels)
        if 'x_phase' not in self.channels:
            self.channels_w_ph['x_phase'] = Channel.ALL
        if 'y_phase' not in self.channels:
            self.channels_w_ph['y_phase'] = Channel.ALL

        # about das phase
        if ch_in == 1:  # Single model can't use das phase
            self.use_das_phase = False
        if self.feature == 'SIV' or self.feature == 'mulspec':
            self.folder_das_phase = 'das-phase'
        elif self.feature == 'DV' or self.feature == 'mulspec4':
            self.folder_das_phase = 'das4-phase'
        else:
            self.use_das_phase = False

    @staticmethod
    def is_featurefile(f: os.DirEntry) -> bool:
        return (f.name.endswith('.npz')
                and not f.name.startswith('metadata')
                and not f.name.startswith('normconst'))

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True) -> Namespace:
        def set_attr_to_parsed(obj: Any, attr_name: str, attr_type: type, parsed: str):
            if parsed == '':
                return
            try:
                v = eval(parsed)
            except:
                v = None
            if attr_type == str or v is None or type(v) != attr_type:
                if (parsed.startswith("'") and parsed.endswith("'")
                        or parsed.startswith('"') and parsed.endswith('"')):
                    parsed = parsed[1:-1]
                if isinstance(obj, dict):
                    obj[attr_name] = parsed
                else:
                    setattr(obj, attr_name, parsed)
            else:
                if isinstance(obj, dict):
                    obj[attr_name] = v
                else:
                    setattr(obj, attr_name, v)

        if not parser:
            parser = ArgumentParser()
        args_already_added = [a.dest for a in parser._actions]
        dict_self = asdict(self)
        for k in dict_self:
            if hasattr(args_already_added, k):
                continue
            if isinstance(dict_self[k], dict):
                for sub_k in dict_self[k]:
                    parser.add_argument(f'--{k}--{sub_k}', default='')
            else:
                parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            if isinstance(dict_self[k], dict):
                for sub_k, sub_v in dict_self[k].items():
                    parsed = getattr(args, f'{k}__{sub_k}')
                    set_attr_to_parsed(getattr(self, k), sub_k, type(sub_v), parsed)
            else:
                parsed = getattr(args, k)
                set_attr_to_parsed(self, k, type(dict_self[k]), parsed)

        self.init_dependent_vars()
        if print_argument:
            print(repr(self))

        return args

    def __repr__(self):
        result = ('-------------------------\n'
                  'Hyper Parameter Settings\n'
                  '-------------------------\n')

        result += '\n'.join(
            [f'{k}: {v}' for k, v in asdict(self).items() if not isinstance(v, ndarray)])
        result += '\n-------------------------'
        return result


hp = _HyperParameters()
