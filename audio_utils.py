from typing import Sequence, Dict, Tuple

import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
import colorsys
from itertools import product as iterprod

from hparams import hp
from utils import TensArr, DICT_PACKAGE, convert
from matlab_lib import Evaluation as EvalModule

EVAL_METRICS = EvalModule.metrics


class LogModule:
    """ make directional spectrograms and multi-channel spectrograms to log scale
        and make log-scaled version to linear scale.

    """
    _KWARGS_SUM = {np: dict(keepdims=True), torch: dict(keepdim=True)}
    eps = hp.eps_for_log
    eps_pow = hp.eps_for_log ** 2

    @classmethod
    def log(cls, a: TensArr) -> TensArr:
        """

        :param a: ... x C
        :return:
        """
        pkg = DICT_PACKAGE[type(a)]

        if a.shape[-1] == 4:  # directional feature + spectrogram
            b = pkg.empty_like(a)
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps_pow) / cls.eps_pow)
            b[..., :3] = a[..., :3] * log_r / (r + cls.eps_pow)
            b[..., 3] = pkg.log10((a[..., 3] + cls.eps) / cls.eps)
        elif a.shape[-1] == 3:  # directional feature
            r = (a**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps_pow) / cls.eps_pow)
            b = a * log_r / (r + cls.eps_pow)
        else:  # spectrogram or multichannel spectrograms
            b = pkg.log10((a + cls.eps) / cls.eps)
        return b

    @classmethod
    def log_(cls, a: TensArr) -> TensArr:
        """

        :param a: ... x C
        :return:
        """
        pkg = DICT_PACKAGE[type(a)]

        if 3 <= a.shape[-1] <= 4:  # if directional feature exists
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps_pow) / cls.eps_pow)
            a[..., :3] *= log_r
            a[..., :3] /= (r + cls.eps_pow)
            if a.shape[-1] == 4:  # if spectrogram exists
                pkg.log10((a[..., 3] + cls.eps) / cls.eps, out=a[..., 3])
        else:  # spectrogram or multichannel spectrograms
            pkg.log10((a + cls.eps) / cls.eps, out=a)
        return a

    @classmethod
    def exp(cls, a: TensArr) -> TensArr:
        """

        :param a: ... x C
        :return:
        """
        pkg = DICT_PACKAGE[type(a)]

        if a.shape[-1] == 4:  # directional feature + spectrogram
            b = pkg.empty_like(a)
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps_pow * (10.**r - 1)
            b[..., :3] = a[..., :3] * exp_r / (r + cls.eps_pow)
            b[..., 3] = cls.eps * (10.**a[..., 3] - 1)
        elif a.shape[-1] == 3:  # directional feature
            r = (a**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps_pow * (10.**r - 1)
            b = a * exp_r / (r + cls.eps_pow)
        else:  # spectrogram or multichannel spectrograms
            b = cls.eps * (10.**a - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr) -> TensArr:
        """

        :param a: ... x C
        :return:
        """
        pkg = DICT_PACKAGE[type(a)]

        if 3 <= a.shape[-1] <= 4:  # if directional feature exists
            r = (a[..., :3]**2).sum(-1, **cls._KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps_pow * (10.**r - 1)
            a[..., :3] *= exp_r
            a[..., :3] /= (r + cls.eps_pow)
            if a.shape[-1] == 4:  # if spectrogram exists
                a[..., 3] = cls.eps * (10.**a[..., 3] - 1)
        else:  # spectrogram or multichannel spectrograms
            a = cls.eps * (10.**a - 1)
        return a


def calc_snrseg(spec_clean: ndarray, spec_est: ndarray, T_ys: Sequence[int] = (0,)) \
        -> float:
    """ calculate segmental SNR in the frequency domain. spec can be a batch.

    :param spec_clean: (B x ) F x T x 1
    :param spec_est: (B x ) F x T x 1
    :param T_ys:
    :return:
    """

    _LIM_UPPER = 35. / 10.  # clip at 35 dB
    _LIM_LOWER = -10. / 10.  # clip at -10 dB
    if len(T_ys) == 1 and spec_clean.shape[0] != 1:
        if T_ys == (0,):
            T_ys = (spec_clean.shape[0],)
        spec_clean = spec_clean[np.newaxis, ...]
        spec_est = spec_est[np.newaxis, ...]
    
    spec_clean = np.abs(spec_clean)
    spec_est = np.abs(spec_est)

    sum_result = np.float32(0.)
    for T, item_clean, item_est in zip(T_ys, spec_clean, spec_est):
        # T
        norm_clean = np.einsum(
            'ftc,ftc->t', item_clean[:, :T, :], item_clean[:, :T, :]
        )
        err = item_est[:, :T, :] - item_clean[:, :T, :]
        norm_err = np.einsum(
            'ftc,ftc->t', err, err
        ) + np.finfo(np.float32).eps

        snrseg = np.log10(norm_clean / norm_err + np.finfo(np.float32).eps)
        np.clip(snrseg, _LIM_LOWER, _LIM_UPPER, out=snrseg)
        sum_result += snrseg.mean()
    sum_result *= 10

    return sum_result


def calc_using_eval_module(y_clean: ndarray, y_est: ndarray,
                           T_ys: Sequence[int] = (0,)) -> Dict[str, float]:
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

    if len(T_ys) > 1:
        metrics = None
        sum_result = None
        for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
            # noinspection PyArgumentList
            metrics, result = EvalModule(item_clean[:T], item_est[:T], hp.fs)
            result = np.array(result)
            if sum_result is None:
                sum_result = result
            else:
                sum_result += result
        sum_result = sum_result.tolist()
    else:
        # noinspection PyArgumentList
        metrics, sum_result = EvalModule(y_clean[0, :T_ys[0]], y_est[0, :T_ys[0]], hp.fs)

    return {k: v for k, v in zip(metrics, sum_result)}


def reconstruct_wave(*args: ndarray, 
                     n_iter=0, momentum=0., n_sample=-1,
                     **kwargs_istft) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag, phase) or (complex spectrogram,) or (mag,)
    :param n_iter: no. of iteration of griffin-lim. 0 for not using griffin-lim.
    :param momentum: fast griffin-lim algorithm momentum
    :param n_sample: number of time samples of output wave
    :param kwargs_istft: kwargs for librosa.istft
    :return:
    """

    if len(args) == 1:
        if np.iscomplexobj(args[0]):
            spec = args[0].squeeze()
            mag = None
            phase = None
        else:
            spec = None
            mag = args[0].squeeze()
            # random initial phase
            phase = np.exp(2j * np.pi * np.random.rand(*mag.shape).astype(mag.dtype))
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError
    if not kwargs_istft:
        kwargs_istft = hp.kwargs_istft
        kwargs_stft = hp.kwargs_stft
    else:
        kwargs_stft = dict(n_fft=hp.n_fft, **kwargs_istft)

    spec_prev = 0
    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.istft(mag * np.exp(1j * phase), **kwargs_istft)
        spec_new = librosa.stft(wave, **kwargs_stft)

        phase = np.angle(spec_new - (momentum / (1 + momentum)) * spec_prev)
        spec_prev = spec_new

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.istft(spec, **kwargs_istft, **kwarg_len)

    return wave


def draw_spectrogram(data: TensArr, to_db=True, show=False, dpi=150, **kwargs):
    if to_db:
        data[data == 0] = data[data > 0].min()
        data = LogModule.log(data)
        data *= 20
    data = data.squeeze()
    data = convert(data, astype=ndarray)

    fig, ax = plt.subplots(dpi=dpi,)
    ax.imshow(data,
              cmap=plt.get_cmap('CMRmap'),
              extent=(0, data.shape[1], 0, hp.fs // 2 // 1000),
              origin='lower', aspect='auto', **kwargs)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (kHz)')
    fig.colorbar(ax.images[0])
    if show:
        fig.show()

    return fig


def cart2sph(x: ndarray, y: ndarray, z: ndarray) -> Tuple[ndarray]:
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return r, az, el


def vec2hsv(cart: ndarray, log_eps=1e-10, r_max=0) -> ndarray:
    """ map 3D cartesian vector to spherical hsv color space.
        The length is scaled in logarithmic scale.

    :param cart: (... x 3)
    :param log_eps: prevent singularity
    :param r_max:
    """
    r, az, el = cart2sph(cart[..., 0], cart[..., 1], cart[..., 2])
    az[az < 0] += 2 * np.pi
    if log_eps != 0:
        r = np.log10((r + log_eps) / log_eps)
    r /= r.max() if r_max == 0 else r_max
    az /= 2 * np.pi
    el /= np.pi

    rgb = np.empty_like(cart)
    shape = cart[..., 0].shape
    for i_flat in range(np.prod(shape)):
        i, j = np.unravel_index(i_flat, shape)
        rgb[i, j] = colorsys.hsv_to_rgb(az[i, j], el[i, j], r[i, j])

    return rgb
