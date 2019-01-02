import matlab.engine
import os
from collections import OrderedDict as ODict
from typing import Sequence, Union, Tuple

import deepdish as dd
import librosa
import numpy as np
import scipy.io as scio
from scipy.linalg import toeplitz
import torch
from matplotlib import pyplot as plt

from normalize import LogInterface as LogModule
import generic as gen
import mypath
from matlab_lib import PESQ_STOI
from pystoi.stoi import stoi
from utils import arr2str


# ---------manually selected---------
N_GRIFFIN_LIM = 20

metadata = dd.io.load(os.path.join(mypath.path('iv_train'), 'metadata.h5'))
# all_files = metadata['path_wavfiles']
L_hop = int(metadata['L_hop'])
N_freq = int(metadata['N_freq'])
N_LOC = int(metadata['N_LOC'])
Fs = int(metadata['Fs'])
N_fft = L_hop * 2

# STFT/iSTFT arguments
kwargs_common = dict(hop_length=L_hop, window='hann', center=True)
kwargs_stft = dict(**kwargs_common, n_fft=N_fft, dtype=np.complex128)
kwargs_istft = dict(**kwargs_common, dtype=np.float64)

del metadata

# bEQspec
DIR_DATA = mypath.path('root')
sft_dict = scio.loadmat(
    os.path.join(DIR_DATA, 'sft_data.mat'),
    variable_names=('bEQf',),
    squeeze_me=True
)
bEQf0 = sft_dict['bEQf'][:, 0][:, np.newaxis, np.newaxis]  # F, T, C
bEQf0_mag = np.abs(bEQf0)
bEQf0_angle = np.angle(bEQf0)
# bEQspec0 = torch.tensor(bEQspec0, dtype=torch.float32, device=OUT_DEVICE)
del sft_dict, bEQf0


# class SNRseg(nn.Module):
#     EINEXPR = 'ftc,ftc->t'
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, y_clean: torch.Tensor, y_est: torch.Tensor,
#                 T_ys: np.ndarray) -> torch.Tensor:
#         if not T_ys:
#             T_ys = (y_est.shape[-2],) * y_est.shape[0]
#         sum_result = torch.zeros(1, device=y_est.device)
#         for i_b, (T, item_clean, item_est) in enumerate(zip(T_ys, y_clean, y_est)):
#             # T
#             norm_y = torch.einsum(SNRseg.einexpr,
#                                   [item_clean[:, :T, :]] * 2)
#             norm_err = torch.einsum(SNRseg.einexpr,
#                                     [item_est[:, :T, :] - item_clean[:, :T, :]]*2)
#
#             sum_result += torch.log10(norm_y / norm_err).mean(dim=1)
#         sum_result *= 10
#         return sum_result


class Measurement:
    __slots__ = ('__len',
                 '__METRICS',
                 '__sum_values',
                 )

    DICT_CALC = dict(SNRseg='Measurement.calc_snrseg',
                     # STOI=Measurement.calc_stoi,
                     PESQ_STOI='Measurement.calc_pesq_stoi',
                     )

    pesq_stoi_module = PESQ_STOI()

    @staticmethod
    def calc_snrseg(y_clean: np.ndarray, y_est: np.ndarray,
                    T_ys: Sequence[int] = (1,)) -> np.ndarray:
        LIM_UPPER = 35. / 10.  # clip at 35 dB
        LIM_LOWER = -10. / 10.  # clip at -10 dB
        if len(T_ys) == 1 and y_clean.shape[0] != 1:
            y_clean = y_clean[np.newaxis, ...]
            y_est = y_est[np.newaxis, ...]

        sum_result = 0.
        for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
            # T
            norm_clean = np.einsum(
                'ftc,ftc->t',
                *([item_clean[:, :T, :]] * 2)
            )
            norm_err = np.einsum(
                'ftc,ftc->t',
                *([item_est[:, :T, :] - item_clean[:, :T, :]] * 2)
            ) + np.finfo(np.float64).eps

            snrseg = np.log10(norm_clean / norm_err + 1e-10)
            np.minimum(snrseg, LIM_UPPER, out=snrseg)
            np.maximum(snrseg, LIM_LOWER, out=snrseg)
            sum_result += snrseg.mean()
        sum_result *= 10

        return sum_result

    @classmethod
    def calc_pesq_stoi(cls, y_clean: np.ndarray, y_est: np.ndarray,
                       *args) -> Tuple[np.float64, np.float64]:
        if y_clean.ndim == 1:
            y_clean = y_clean[np.newaxis, ...]
            y_est = y_est[np.newaxis, ...]

        sum_result = np.zeros(2)
        for item_clean, item_est in zip(y_clean, y_est):
            temp = cls.pesq_stoi_module(item_clean, item_est, Fs)
            sum_result += np.array([temp['PESQ'], temp['STOI']])

        return sum_result[0], sum_result[1]

    @staticmethod
    def calc_stoi(y_clean: np.ndarray, y_est: np.ndarray):
        sum_result = 0.
        for item_clean, item_est in zip(y_clean, y_est):
            sum_result += stoi(item_clean, item_est, Fs)
        return sum_result

    def __init__(self, *metrics):
        self.__len = 0
        self.__METRICS = metrics
        # self.__seq_values = {metric: torch.empty(max_size) for metric in self.__METRICS}
        self.__sum_values: ODict = None

    def __len__(self):
        return self.__len

    def __str__(self):
        return self._str_measure(self.average())

    # def __getitem__(self, idx):
    #     return [self.__seq_values[metric][idx] for metric in self.__METRICS]

    def average(self) -> 'ODict[str, torch.Tensor]':
        return ODict([(metric, sum_ / self.__len)
                      for metric, sum_ in self.__sum_values.items()])

    def append(self, y: np.ndarray, out: np.ndarray,
               T_ys: Union[int, Sequence[int]]) -> str:
        values = ODict([(metric, eval(self.DICT_CALC[metric])(y, out, T_ys))
                        for metric in self.__METRICS])
        if self.__len:
            for metric, v in values.items():
                # self.__seq_values[metric][self.__len] = self.DICT_CALC[metric](y, out, T_ys)
                self.__sum_values[metric] += v
        else:
            self.__sum_values = values

        self.__len += len(T_ys) if hasattr(T_ys, '__len__') else 1
        return self._str_measure(values)

    @staticmethod
    def _str_measure(values: ODict) -> str:
        return '\t'.join(
            [f"{metric}={arr2str(v, 'f')} " for metric, v in values.items()]
        )

    # stft_module = STFT(N_fft, L_hop)
    #
    # if USE_CUDA:
    #     stft_module = nn.DataParallel(stft_module).cuda()

    # measure_mag = Measurement('SNRseg')
    # measure_wav = Measurement('PESQ_STOI')


def reconstruct_wave(power: np.ndarray, phase: np.ndarray,
                     *, n_sample=-1, do_griffin_lim=False) -> np.ndarray:
    power = power.squeeze()
    phase = phase.squeeze()

    # power to mag
    mag = np.sqrt(power, out=power)

    if do_griffin_lim:
        wave = griffin_lim(mag, phase, n_sample=n_sample, n_iter=N_GRIFFIN_LIM)
    else:
        wave = librosa.core.istft(mag * np.exp(1j * phase), **kwargs_istft)
        if n_sample != -1:
            wave = wave[:n_sample]

    max_amplitude = np.max(np.abs(wave))
    if max_amplitude > 1:
        wave /= max_amplitude
        print(f'wave is scaled by {max_amplitude} to prevent clipping.')

    return wave


def griffin_lim(mag: np.ndarray, phase: np.ndarray,
                *, n_iter: int, n_sample=-1) -> np.ndarray:
    # wave = scsig.istft(mag*np.exp(1j*phase), **kwargs_stft)[-1]
    for _ in range(n_iter - 1):
        wave = librosa.core.istft(mag * np.exp(1j * phase), **kwargs_istft)

        spec = librosa.core.stft(wave, **kwargs_stft)
        phase = np.angle(spec)

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    wave = librosa.core.istft(mag * np.exp(1j * phase), **kwargs_istft, **kwarg_len)
    # wave = stft_module.inverse(mag, phase)[..., :n_sample]
    # for _ in range(n_iter - 1):
    #     _, phase = stft_module.transform(wave)
    #     wave = stft_module.inverse(mag, phase)[..., :n_sample]
    return wave


def delta(*data: gen.TensArr, axis: int, L=2) -> gen.TensArrOrSeq:
    dim = gen.ndim(data[0])
    dtype = gen.convert_dtype(data[0].dtype, np)
    if axis < 0:
        axis += dim

    max_len = max([item.shape[axis] for item in data])

    # Einsum expression
    # ex) if the member of a has the dim (b,c,f,t), (thus, axis=3)
    # einxp: ij,abcd -> abci
    str_axes = ''.join([chr(ord('a') + i) for i in range(dim)])
    str_new_axes = ''.join([chr(ord('a') + i) if i != axis else 'i'
                            for i in range(dim)])
    ein_expr = f'ij,{str_axes}->{str_new_axes}'

    # Create Toeplitz Matrix (T-2L, T)
    col = np.zeros(max_len - 2 * L, dtype=dtype)
    col[0] = -L

    row = np.zeros(max_len, dtype=dtype)
    row[:2 * L + 1] = range(-L, L + 1)

    denominator = np.sum([ll**2 for ll in range(1, L + 1)])
    tplz_mat = toeplitz(col, row) / (2 * denominator)

    # Convert to Tensor
    if type(data[0]) == torch.Tensor:
        if data[0].device == torch.device('cpu'):
            tplz_mat = torch.from_numpy(tplz_mat)
        else:
            tplz_mat = torch.tensor(tplz_mat, device=data[0].device)

    # Calculate
    result = [type(data[0])] * len(data)
    for idx, item in enumerate(data):
        length = item.shape[axis]
        result[idx] = gen.einsum(ein_expr,
                                 (tplz_mat[:length - 2 * L, :length], item))

    return result if len(result) > 1 else result[0]


def draw_spectrogram(data: gen.TensArr, power=True, show=False):
    scale_factor = 10 if power else 20
    data = LogModule.log(data)
    data = data.squeeze()
    data *= scale_factor
    data = gen.convert(data, astype=np.ndarray)

    fig = plt.figure()
    plt.imshow(data,
               cmap=plt.get_cmap('CMRmap'),
               extent=(0, data.shape[1], 0, Fs//2),
               origin='lower', aspect='auto')
    plt.xlabel('Frame Index')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    if show:
        plt.show()

    return fig


def bnkr_equalize_(mag, phase=None):
    mag *= bEQf0_mag
    if phase is not None:
        phase += bEQf0_angle
        return mag, phase
    else:
        return mag
