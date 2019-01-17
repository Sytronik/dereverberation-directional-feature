from collections import OrderedDict as ODict
from typing import Sequence

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz

import config as cfg
import generic as gen
from matlab_lib import Evaluation
from normalize import LogInterface as LogModule
from utils import static_vars


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

# class Measurement:
#     __slots__ = ('__len',
#                  '__METRICS',
#                  '__sum_values',
#                  )
#
#     DICT_CALC = dict(SNRseg='Measurement.calc_snrseg',
#                      # STOI=Measurement.calc_stoi,
#                      PESQ_STOI='Measurement.calc_pesq_stoi',
#                      )
#
#     def __init__(self, *metrics):
#         self.__len = 0
#         self.__METRICS = metrics
#         # self.__seq_values = {metric: torch.empty(max_size) for metric in self.__METRICS}
#         self.__sum_values: ODict = None
#
#     def __len__(self):
#         return self.__len
#
#     def __str__(self):
#         return self._str_measure(self.average())
#
#     # def __getitem__(self, idx):
#     #     return [self.__seq_values[metric][idx] for metric in self.__METRICS]
#
#     def average(self):
#         """
#
#         :rtype: OrderedDict[str, torch.Tensor]
#         """
#         return ODict([(metric, sum_ / self.__len)
#                       for metric, sum_ in self.__sum_values.items()])
#
#     def append(self, y: np.ndarray, out: np.ndarray,
#                T_ys: Union[int, Sequence[int]]) -> str:
#         values = ODict([(metric, eval(self.DICT_CALC[metric])(y, out, T_ys))
#                         for metric in self.__METRICS])
#         if self.__len:
#             for metric, v in values.items():
#                 # self.__seq_values[metric][self.__len] = self.DICT_CALC[metric](y, out, T_ys)
#                 self.__sum_values[metric] += v
#         else:
#             self.__sum_values = values
#
#         self.__len += len(T_ys) if hasattr(T_ys, '__len__') else 1
#         return self._str_measure(values)
#
#     @staticmethod
#     def _str_measure(values: ODict) -> str:
#         return '\t'.join(
#             [f"{metric}={arr2str(v, 'f')} " for metric, v in values.items()]
#         )
#
#
# def calc_stoi(y_clean: np.ndarray, y_est: np.ndarray):
#     sum_result = 0.
#     for item_clean, item_est in zip(y_clean, y_est):
#         sum_result += stoi(item_clean, item_est, cfg.Fs)
#     return sum_result


def calc_snrseg(y_clean: np.ndarray, y_est: np.ndarray,
                T_ys: Sequence[int] = (0,)) -> np.ndarray:
    _LIM_UPPER = 35. / 10.  # clip at 35 dB
    _LIM_LOWER = -10. / 10.  # clip at -10 dB
    if len(T_ys) == 1 and y_clean.shape[0] != 1:
        T_ys = (y_clean.shape[0],)
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]

    sum_result = 0.
    for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
        # T
        norm_clean = np.einsum(
            'ftc,ftc->t',
            *((item_clean[:, :T, :],) * 2)
        )
        norm_err = np.einsum(
            'ftc,ftc->t',
            *((item_est[:, :T, :] - item_clean[:, :T, :],) * 2)
        ) + np.finfo(np.float64).eps

        snrseg = np.log10(norm_clean / norm_err + np.finfo(np.float64).eps)
        np.minimum(snrseg, _LIM_UPPER, out=snrseg)
        np.maximum(snrseg, _LIM_LOWER, out=snrseg)
        sum_result += snrseg.mean()
    sum_result *= 10

    return sum_result


@static_vars(module=None)
def calc_using_eval_module(y_clean: np.ndarray, y_est: np.ndarray,
                           T_ys: Sequence[int] = (0,)) -> ODict:
    if not calc_using_eval_module.module:
        calc_using_eval_module.module = Evaluation()
    if y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],)*y_clean.shape[0]

    keys = None
    sum_result = None
    for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
        temp = calc_using_eval_module.module(item_clean[:T], item_est[:T], cfg.Fs)
        result = np.array(list(temp.values()))
        if not keys:
            keys = temp.keys()
            sum_result = result
        else:
            sum_result += result

    return ODict(zip(keys, sum_result.tolist()))


def wave_scale_fix(wave: np.ndarray, amp_limit=1., message='') -> np.ndarray:
    max_amp = np.max(np.abs(wave))
    if max_amp > amp_limit:
        wave /= max_amp
        if message:
            print(f'{message} is scaled by {max_amp / amp_limit} to prevent clipping.')

    return wave


def reconstruct_wave(mag: np.ndarray, phase: np.ndarray,
                     *, n_iter=0, n_sample=-1) -> np.ndarray:
    mag = mag.squeeze()
    phase = phase.squeeze()

    for _ in range(n_iter - 1):
        wave = librosa.core.istft(mag * np.exp(1j * phase), **cfg.KWARGS_ISTFT)

        spec = librosa.core.stft(wave, **cfg.KWARGS_STFT)
        phase = np.angle(spec)

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    wave = librosa.core.istft(mag * np.exp(1j * phase), **cfg.KWARGS_ISTFT, **kwarg_len)

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


def draw_spectrogram(data: gen.TensArr, is_power=False, show=False, **kwargs):
    """
    
    :param data: 
    :param is_power: 
    :param show: 
    :param kwargs: vmin, vmax
    :return: 
    """

    scale_factor = 10 if is_power else 20
    data = LogModule.log(data)
    data = data.squeeze()
    data *= scale_factor
    data = gen.convert(data, astype=np.ndarray)

    fig = plt.figure()
    plt.imshow(data,
               cmap=plt.get_cmap('CMRmap'),
               extent=(0, data.shape[1], 0, cfg.Fs//2),
               origin='lower', aspect='auto', **kwargs)
    plt.xlabel('Frame Index')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    if show:
        plt.show()

    return fig


def bnkr_equalize_(mag, phase=None):
    assert mag.shape[0] == cfg.bEQf0_mag.shape[0]
    bEQf0_mag = cfg.bEQf0_mag
    while mag.ndim < bEQf0_mag.ndim:
        bEQf0_mag = bEQf0_mag[..., 0]

    mag *= bEQf0_mag

    if phase is not None:
        assert phase.shape[0] == mag.shape[0]
        bEQf0_angle = cfg.bEQf0_angle
        while phase.ndim < bEQf0_mag.ndim:
            bEQf0_angle = bEQf0_angle[..., 0]

        phase += bEQf0_angle
        return mag, phase
    else:
        return mag
