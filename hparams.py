import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import scipy.io as scio
# noinspection PyCompatibility
from dataclasses import asdict, dataclass, field
from numpy import ndarray


# noinspection PyArgumentList
class Channel(Enum):
    ALL = slice(None)
    MAG = slice(-1, None)
    NONE = None


@dataclass
class _HyperParameters:
    # devices
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    out_device: Union[int, str] = 3
    num_disk_workers: int = 3

    # select dataset
    DF: str = 'IV'
    # DF: str = 'DirAC
    room_train: str = 'room1'
    room_test: str = 'room1'
    room_create: str = ''

    method: str = 'mag'
    # method: str = 'complex'
    # method: str = 'magphase'
    # method: str = 'magbpd'
    model_name: str = 'UNet'

    # dirspec parameters
    fs: int = 16000
    n_fft: int = 512
    l_frame: int = 512
    n_freq: int = 257
    l_hop: int = 256

    # log & normalize
    eps_for_log: float = 1e-10
    use_log: bool = True
    refresh_const: bool = False

    # training
    n_data_per_room: int = 13 * 300
    n_test_per_room: int = 10 * 100
    train_ratio: float = 0.77
    n_epochs: int = 210
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5  # Adam weight_decay
    weight_loss: tuple = (0.1, 1)  # complex
    # weight_loss: tuple = (0.3, 1, 0.08)  # magphase

    # reconstruction
    use_glim: bool = True
    n_glim_iter: int = 20
    do_bnkr_eq: bool = False

    # paths
    logdir: str = f'./result/test'
    path_speech: Path = Path('./data/TIMIT')
    path_feature: Path = Path('./backup')
    form_path_normconst: str = 'normconst_{}.npz'
    s_path_metadata: str = ''

    # file names
    form_feature: str = '{:05d}_{:04d}_{}_{:02d}.npz'
    form_result: str = 'dirspec_{}.mat'
    log_fname: str = 'log.txt'
    scalars_fname: str = 'scalars.json'
    hparams_fname: str = 'hparams.txt'

    channels: Dict[str, Channel] = field(init=False)
    UNet: Dict[str, Any] = field(init=False)
    scheduler: Dict[str, Any] = field(init=False)
    spec_data_names: Dict[str, str] = field(init=False)

    dummy_input_size: Tuple = None
    dict_path: Dict[str, Path] = None
    kwargs_stft: Dict[str, Any] = None
    kwargs_istft: Dict[str, Any] = None
    n_loc: Dict[str, int] = None
    n_loss_term: int = None
    keys_trannorm: Sequence[Tuple] = None
    period_save_state: int = None
    channels_w_ph: Dict[str, Channel] = None
    bnkr_inv0: ndarray = None
    bnkr_inv0_mag: ndarray = None
    bnkr_inv0_ph: ndarray = None

    def __post_init__(self):
        self.channels = dict(speech_fname=Channel.NONE,
                             x=Channel.ALL,
                             # x=Channel.MAG,
                             y=Channel.MAG,
                             # x_phase=Channel.ALL,
                             # y_phase=Channel.ALL,
                             )

        self.UNet = dict(ch_base=32,
                         depth=4,
                         use_cbam=False,
                         )
        self.scheduler = dict(restart_period=10,
                              t_mult=2,
                              eta_threshold=1.5,
                              )

        self.spec_data_names = dict(x='dirspec_room', y='dirspec_free',
                                    x_phase='phase_room', y_phase='phase_free',
                                    x_bpd='bpd_room', y_bpd='bpd_free',
                                    speech_fname='speech_fname',
                                    out='dirspec_estimated',
                                    out_phase='phase_estimated',
                                    out_bpd='bpd_estimated',
                                    )

    def init_dependent_vars(self):
        self.logdir = Path(self.logdir)
        # nn
        ch_in = 4 if self.channels['x'] == Channel.ALL else 1
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
            RIR_Ys=self.path_feature / f'RIR_Ys_TRAIN13_TEST10_{self.room_create}.mat',

            speech_train=self.path_speech / 'TRAIN',
            speech_test=self.path_speech / 'TEST',

            feature_train=path_feature_train,
            feature_seen=path_feature_test / 'SEEN',
            feature_unseen=path_feature_test / 'UNSEEN',

            form_normconst_train=str(path_feature_train / self.form_path_normconst),
            form_normconst_seen=str(path_feature_test / self.form_path_normconst),
            form_normconst_unseen=str(path_feature_test / self.form_path_normconst),

            figures=Path('./figures'),
        )

        # dirspec parameters
        self.kwargs_stft = dict(hop_length=self.l_hop, window='hann', center=True,
                                n_fft=self.n_fft, dtype=np.complex64)
        self.kwargs_istft = dict(hop_length=self.l_hop, window='hann', center=True,
                                 dtype=np.float32)
        self.n_loc = dict()
        for kind in ('train', 'seen', 'unseen'):
            path_metadata = self.dict_path[f'feature_{kind}'] / 'metadata.mat'
            if path_metadata.exists():
                self.n_loc[kind] = scio.loadmat(
                    str(path_metadata), variable_names=('n_loc',)
                )['n_loc'].item()
            else:
                print(f'n_loc of "{kind}" not loaded.')

        # log & normalize
        key_trannorm = (self.method, 'meanstd', self.use_log)
        if self.method == 'mag' or self.method == 'magbpd':
            self.n_loss_term = 1
            self.keys_trannorm = (key_trannorm,)
        elif self.method == 'complex':
            self.n_loss_term = 2
            self.keys_trannorm = (key_trannorm, ('mag', 'meanstd', True))
        elif self.method == 'magphase':
            self.n_loss_term = 3
            self.keys_trannorm = (('mag', 'meanstd', True), ('complex', 'meanstd', False))

        # training
        if not self.period_save_state:
            self.period_save_state = self.scheduler['restart_period'] // 2

        # reconstruction
        self.channels_w_ph = dict(**self.channels)
        if 'x_phase' not in self.channels:
            self.channels_w_ph['x_phase'] = Channel.ALL

        if 'y_phase' not in self.channels:
            self.channels_w_ph['y_phase'] = Channel.ALL

        # sft_dict = scio.loadmat(str(self.dict_path['sft_data']),
        #                         variable_names=('bEQf',),
        #                         squeeze_me=True)
        # self.bnkr_inv0 = sft_dict['bEQf'][:, 0]
        # self.bnkr_inv0 = self.bnkr_inv0[:, np.newaxis, np.newaxis]  # F, T(1), C(1)
        # self.bnkr_inv0_mag = np.abs(self.bnkr_inv0)
        # self.bnkr_inv0_ph = np.angle(self.bnkr_inv0)
        # if self.DF == 'DirAC':
        #     self.do_bnkr_eq = False
        # else:
        #     self.do_bnkr_eq = True

    @staticmethod
    def is_featurefile(f: os.DirEntry) -> bool:
        return (f.name.endswith('.npz')
                and not f.name.startswith('metadata')
                and not f.name.startswith('normconst_'))

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True) -> Namespace:
        if not parser:
            parser = ArgumentParser()
        args_already_added = [a.dest for a in parser._actions]
        dict_self = asdict(self)
        for k in dict_self:
            if hasattr(args_already_added, k):
                continue
            parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            parsed = getattr(args, k)
            if parsed == '':
                continue
            if isinstance(dict_self[k], str):
                if (parsed.startswith("'") and parsed.endwith("'")
                        or parsed.startswith('"') and parsed.endwith('"')):
                    parsed = parsed[1:-1]
                setattr(self, k, parsed)
            else:
                v = eval(parsed)
                if isinstance(v, type(dict_self[k])):
                    setattr(self, k, eval(parsed))

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


hp = _HyperParameters()