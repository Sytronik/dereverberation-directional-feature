from collections import OrderedDict as ODict
from typing import Sequence

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import ndarray

from hparams import hp
import generic as gen
from generic import TensArr
from matlab_lib import Evaluation as EvalModule


EVAL_METRICS = EvalModule.metrics


class LogModule:
    _KWARGS_SUM = {np: dict(keepdims=True), torch: dict(keepdim=True)}
    eps = hp.eps_for_log

    @classmethod
    def log(cls, a: TensArr) -> TensArr:
        """

        :param a: * x C, C can be 1, 3, 4
        :return:
        """
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:  # directional feature + spectrogram
            b = pkg.empty_like(a)
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps) / cls.eps)
            b[..., :3] = a[..., :3] * log_r / (r + cls.eps)
            b[..., 3] = pkg.log10((a[..., 3] + cls.eps) / cls.eps)
        elif a.shape[-1] == 3:  # directional feature
            r = (a**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps) / cls.eps)
            b = a * log_r / (r + cls.eps)
        else:  # spectrogram
            b = pkg.log10((a + cls.eps) / cls.eps)
        return b

    @classmethod
    def log_(cls, a: TensArr) -> TensArr:
        """

        :param a: * x C, C can be 1, 3, 4
        :return:
        """
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] >= 3:  # if directional feature exists
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps) / cls.eps)
            a[..., :3] *= log_r
            a[..., :3] /= (r + cls.eps)
            if a.shape[-1] == 4:  # if spectrogram exists
                a[..., 3] = pkg.log10((a[..., 3] + cls.eps) / cls.eps)
        else:  # spectrogram
            pkg.log10((a + cls.eps) / cls.eps, out=a)
        return a

    @classmethod
    def exp(cls, a: TensArr) -> TensArr:
        """

        :param a: * x C, C can be 1, 3, 4
        :return:
        """
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:  # directional feature + spectrogram
            b = pkg.empty_like(a)
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps * (10.**r - 1)
            b[..., :3] = a[..., :3] * exp_r / (r + cls.eps)
            b[..., 3] = cls.eps * (10.**a[..., 3] - 1)
        elif a.shape[-1] == 3:  # directional feature
            r = (a**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps * (10.**r - 1)
            b = a * exp_r / (r + cls.eps)
        else:  # spectrogram
            b = cls.eps * (10.**a - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr) -> TensArr:
        """

        :param a: * x C, C can be 1, 3, 4
        :return:
        """
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] >= 3:  # if directional feature exists
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps * (10.**r - 1)
            a[..., :3] *= exp_r
            a[..., :3] /= (r + cls.eps)
            if a.shape[-1] == 4:  # if spectrogram exists
                a[..., 3] = cls.eps * (10.**a[..., 3] - 1)
        else:  # spectrogram
            a = cls.eps * (10.**a - 1)
        return a


def calc_snrseg(y_clean: ndarray, y_est: ndarray, T_ys: Sequence[int] = (0,)) \
        -> float:
    """ calculate snrseg. y can be a batch.

    :param y_clean:
    :param y_est:
    :param T_ys:
    :return:
    """

    _LIM_UPPER = 35. / 10.  # clip at 35 dB
    _LIM_LOWER = -10. / 10.  # clip at -10 dB
    if len(T_ys) == 1 and y_clean.shape[0] != 1:
        if T_ys == (0,):
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


def calc_using_eval_module(y_clean: ndarray, y_est: ndarray,
                           T_ys: Sequence[int] = (0,)) -> ODict:
    """ calculate metric using EvalModule. y can be a batch.

    :param y_clean:
    :param y_est:
    :param T_ys:
    :return:
    """

    if y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],) * y_clean.shape[0]

    keys = None
    sum_result = None
    for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
        # noinspection PyArgumentList
        temp: ODict = EvalModule(item_clean[:T], item_est[:T], hp.fs)
        result = np.array(list(temp.values()))
        if not keys:
            keys = temp.keys()
            sum_result = result
        else:
            sum_result += result

    return ODict(zip(keys, sum_result.tolist()))


def reconstruct_wave(*args: ndarray, n_iter=0, n_sample=-1) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag_spectrogram, phase_spectrogram) or (complex_spectrogram,)
    :param n_iter: no. of iteration of griffin-lim. 0 for not using griffin-lim.
    :param n_sample: number of samples of output wave
    :return:
    """

    if len(args) == 1:
        spec = args[0].squeeze()
        mag = None
        phase = None
        assert np.iscomplexobj(spec)
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError

    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.core.istft(mag * np.exp(1j * phase), **hp.kwargs_istft)

        phase = np.angle(librosa.core.stft(wave, **hp.kwargs_stft))

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.core.istft(spec, **hp.kwargs_istft, **kwarg_len)

    return wave


def draw_spectrogram(data: TensArr, to_db=True, show=False, dpi=150, **kwargs):
    """
    
    :param data:
    :param to_db:
    :param show:
    :param dpi:
    :param kwargs: vmin, vmax
    :return: 
    """

    if to_db:
        data[data == 0] = data[data > 0].min()
        data = LogModule.log(data)
        data *= 20
    data = data.squeeze()
    data = gen.convert(data, astype=ndarray)

    fig, ax = plt.subplots(dpi=dpi,)
    ax.imshow(data,
               cmap=plt.get_cmap('CMRmap'),
               extent=(0, data.shape[1], 0, hp.fs // 2),
               origin='lower', aspect='auto', **kwargs)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(ax.images[0])
    if show:
        fig.show()

    return fig
