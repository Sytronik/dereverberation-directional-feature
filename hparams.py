import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
# noinspection PyCompatibility
from dataclasses import asdict, dataclass, field
from numpy import ndarray


# noinspection PyArgumentList
class Channel(Enum):
    ALL = slice(None)
    LAST = slice(-1, None)
    NONE = None


@dataclass
class _HyperParameters:
    # devices
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    out_device: Union[int, str] = 2
    num_disk_workers: int = 4

    # select dataset
    DF: str = 'IV'
    # DF: str = 'DirAC'
    # DF: str = 'mulspec'
    room_train: str = 'room1+2+3'
    room_test: str = 'room1+2+3'
    room_create: str = ''

    model_name: str = 'UNet'

    # dirspec parameters
    fs: int = 16000
    n_fft: int = 512
    l_frame: int = 512
    n_freq: int = 257
    l_hop: int = 256
    n_data_per_room: int = 23 * 300
    n_test_per_room: int = 10 * 100

    # log & normalize
    eps_for_log: float = 1e-5
    use_log: bool = True
    refresh_const: bool = False

    # training
    n_data: int = 0  # <=0 to use all data
    train_ratio: float = 0.77
    n_epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4  # Adam weight_decay
    threshold_stop: float = 0.99

    # reconstruction
    n_glim_iter: int = 20  # 0 for not using Griffin Lim

    # summary
    period_save_state: int = 4
    draw_test_fig: bool = False

    # paths
    logdir: str = f'./result/test'  # will be converted to type Path
    path_speech: Path = Path('./data/TIMIT')
    # path_feature: Path = Path('./data')
    path_feature: Path = Path('./backup')
    s_path_metadata: str = ''

    # file names
    form_feature: str = '{:05d}_{:04d}_{}_{:02d}.npz'  # idx, i_speech, room, i_loc
    form_result: str = 'dirspec_{}.mat'

    # defined in __post_init__
    channels: Dict[str, Channel] = field(init=False)
    UNet: Dict[str, Any] = field(init=False)
    scheduler: Dict[str, Any] = field(init=False)
    spec_data_names: Dict[str, str] = field(init=False)

    # dependent variables
    dummy_input_size: Tuple = None
    dict_path: Dict[str, Path] = None
    kwargs_stft: Dict[str, Any] = None
    kwargs_istft: Dict[str, Any] = None
    channels_w_ph: Dict[str, Channel] = None

    def __post_init__(self):
        self.channels = dict(path_speech=Channel.NONE,
                             x=Channel.ALL,
                             # x=Channel.LAST,
                             x_mag=Channel.NONE,
                             y=Channel.LAST,
                             # x_phase=Channel.ALL,
                             # y_phase=Channel.ALL,
                             )

        self.UNet = dict(ch_base=64,
                         depth=4,
                         use_cbam=False,
                         )
        self.scheduler = dict(restart_period=4,
                              t_mult=2,
                              eta_threshold=1.5,
                              )

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
            if self.DF == 'mulspec':
                ch_in = 64
            else:
                ch_in = 4
        else:
            ch_in = 1
        ch_out = 4 if self.channels['y'] == Channel.ALL else 1
        if 'x_phase' in self.channels:
            ch_in += 1
        if 'y_phase' in self.channels:
            ch_out += 1

        self.UNet['ch_in'] = ch_in
        self.UNet['ch_out'] = ch_out

        self.dummy_input_size = (ch_in,
                                 self.n_freq,
                                 int(2**np.floor(np.log2(4 / 3 * self.n_freq))))

        # path
        if self.room_create:
            self.room_train = self.room_create
            self.room_test = self.room_create
        path_feature_train = self.path_feature / f'{self.DF}_{self.room_train}/TRAIN'
        path_feature_test = self.path_feature / f'{self.DF}_{self.room_test}/TEST'
        self.dict_path = dict(
            sft_data=self.path_feature / 'sft_data_32ms.mat',
            RIR_Ys=self.path_feature / f'RIR_Ys_{self.room_create}.mat',

            speech_train=self.path_speech / 'TRAIN',
            speech_test=self.path_speech / 'TEST',

            feature_train=path_feature_train,
            feature_seen=path_feature_test / 'SEEN',
            feature_unseen=path_feature_test / 'UNSEEN',

            normconst_train=path_feature_train / 'normconst.npz',
            normconst_seen=path_feature_test / 'normconst.npz',
            normconst_unseen=path_feature_test / 'normconst.npz',

            figures=Path('./figures'),
        )

        # dirspec parameters
        self.kwargs_stft = dict(hop_length=self.l_hop, window='hann', center=True,
                                n_fft=self.n_fft, dtype=np.complex64)
        self.kwargs_istft = dict(hop_length=self.l_hop, window='hann', center=True,
                                 dtype=np.float32)

        # reconstruction
        self.channels_w_ph = dict(**self.channels)
        if 'x_phase' not in self.channels:
            self.channels_w_ph['x_phase'] = Channel.ALL

        if 'y_phase' not in self.channels:
            self.channels_w_ph['y_phase'] = Channel.ALL

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
