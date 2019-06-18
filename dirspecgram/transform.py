from abc import ABCMeta, abstractmethod
from itertools import islice

import numpy as np
import scipy.signal as scsig
import torch
from numpy import ndarray

from hparams import hp
import generic as gen
from generic import DataPerDevice, TensArr, TensArrOrSeq

_KWARGS_SUM = {np: dict(keepdims=True), torch: dict(keepdim=True)}


def magphase2realimag(a: TensArr, a_phase: TensArr, concat=True) -> TensArrOrSeq:
    pkg = gen.dict_package[type(a)]
    a_real = a[..., -1:] * pkg.cos(a_phase)
    a_imag = a[..., -1:] * pkg.sin(a_phase)
    if concat:
        return gen.cat((a[..., :-1], a_real, a_imag), axis=-1)
    else:
        return gen.cat((a[..., :-1], a_real), axis=-1), a_imag


def realimag2magphase(a: TensArr, concat=True) -> TensArrOrSeq:
    a_mag = (a[..., -2:-1]**2 + a[..., -1:]**2)**0.5
    a_phase = gen.arctan2(a[..., -1:], a[..., -2:-1])
    if concat:
        return gen.cat((a[..., :-2], a_mag, a_phase), axis=-1)
    else:
        return gen.cat((a[..., :-2], a_mag), axis=-1), a_phase


def principle_(angle):
    angle += np.pi
    angle %= (2 * np.pi)
    angle -= np.pi
    return angle


class BPD:
    N_CTXT = 3
    _basebands: DataPerDevice = None
    _win_context: DataPerDevice = None
    _win_spec_fft: ndarray = None

    @classmethod
    def _get_basebands_like(cls, a: TensArr) -> TensArr:
        if not cls._basebands:
            bb = np.array(range(a.shape[-3]))[:, np.newaxis, np.newaxis]
            bb = 2. * np.pi * hp.l_hop * bb / hp.n_fft
            # bb = 2. * np.pi * bb / hp.N_fft
            cls._basebands = DataPerDevice(bb)
        return cls._basebands.get_like(a)

    @classmethod
    def phase2bpd(cls, phase):
        """

        :param phase: (B,) F, T, 1
        :return: (B,) F, T, 1
        """
        pkg = gen.dict_package[type(phase)]

        bpd = pkg.zeros_like(phase)
        bpd[..., :, :-1, :] = phase[..., :, 1:, :] - phase[..., :, :-1, :]

        bpd -= cls._get_basebands_like(phase)
        bpd = principle_(bpd)
        return bpd

    @classmethod
    def phase2bpd_(cls, phase):
        """

        :param phase: (B,) F, T, 1
        :return: (B,) F, T, 1
        """
        pkg = gen.dict_package[type(phase)]

        bpd = phase
        pkg.add(phase[..., :, 1:, :], -phase[..., :, :-1, :],
                out=bpd[..., :, :-1, :])

        if pkg == torch:
            bpd[..., :, -1, :].fill_(0.)
        else:
            bpd[..., :, -1, :].fill(0.)

        bpd -= cls._get_basebands_like(phase)
        bpd = principle_(bpd)
        return bpd

    @classmethod
    def bpd2phase_(cls, bpd: TensArr, mag: TensArr,
                   phase_init: TensArr, mask: TensArr, along_freq=True):
        """

        :param bpd: F, T, 1
        :param mag: F, T, 1
        :param phase_init: F, T, 1
        :param mask: F, T, 1
        :param along_freq:
        :return: F, T, 1
        """
        weight = None
        sum_weight = None

        def weighted_avg(a):
            return (weight * a).sum(1, **_KWARGS_SUM[pkg]) / sum_weight

        pkg = gen.dict_package[type(bpd)]

        bpd += cls._get_basebands_like(bpd)

        # F, T
        if pkg == torch:
            bpd = bpd.squeeze_()
        else:
            bpd = bpd.squeeze()

        ph_diff = principle_(bpd)  # F, T

        # along time axis
        phase = pkg.empty_like(bpd)  # F, T
        if not cls._win_context:
            cls._win_context = DataPerDevice(
                # scsig.windows.hann(cls.N_CTXT * 2 + 3)[1:-1]
                scsig.windows.hamming(cls.N_CTXT * 2 + 1)
            )
        win_context = cls._win_context.get_like(bpd)

        backup = None
        for t1 in range(bpd.shape[1]):
            left = cls.N_CTXT - min(cls.N_CTXT, t1)
            right = cls.N_CTXT + min(cls.N_CTXT + 1, bpd.shape[1] - t1)
            estimates = pkg.zeros(hp.n_freq, len(win_context)).to(bpd)
            for t2 in range(cls.N_CTXT - 1, left - 1, -1):
                estimates[:, t2] = (estimates[:, t2 + 1]
                                    + ph_diff[:, t1 + t2 - cls.N_CTXT])
            for t2 in range(cls.N_CTXT + 1, right):
                estimates[:, t2] = (estimates[:, t2 - 1]
                                    - ph_diff[:, t1 + t2 - cls.N_CTXT - 1])
            estimates[:, left:right] \
                += phase_init[:, t1 + (left - cls.N_CTXT):t1 + (right - cls.N_CTXT), 0]

            weight = (
                    win_context[left:right]
                    * mask[:, t1 + (left - cls.N_CTXT):t1 + (right - cls.N_CTXT), 0]
            )
            sum_weight = weight.sum(1, **_KWARGS_SUM[pkg])

            if backup is None:
                estimates[:, left:right] = principle_(estimates[:, left:right])

                biased = estimates[:, left:right]
                biased[biased < 0] += 2*np.pi
                mean = weighted_avg(estimates[:, left:right])
                mean_biased = weighted_avg(biased)
                var = weighted_avg((estimates[:, left:right] - mean)**2)
                var_biased = weighted_avg((biased - mean_biased)**2)

                select = var[:, 0] <= var_biased[:, 0]
                phase[select, t1] = mean[select, 0]
                phase[select == False, t1] = mean_biased[select == False, 0]
            else:
                estimates[:, left:right] = gen.unwrap(
                    gen.stack((backup[:, left:right], estimates[:, left:right])),
                    axis=0
                )[1]
                phase[:, t1] = weighted_avg(estimates[:, left:right])
            # backup = estimates

        if not along_freq:
            return gen.expand_dims(phase, -1)  # F, T, 1

        # along frequency axis
        if cls._win_spec_fft is None:
            cls._win_spec_fft = scsig.windows.hann(hp.n_fft, sym=False)
            cls._win_spec_fft = np.fft.fft(cls._win_spec_fft)

        mag = gen.convert(mag, astype=np.ndarray).squeeze()  # F, T
        peaks_i = mag[0:1, :] > mag[1:2, :]
        peaks = (mag[:-2, :] < mag[1:-1, :]) & (mag[1:-1, :] > mag[2:, :])
        peaks_f = mag[-1:, :] > mag[-2:-1, :]
        peaks = np.concatenate((peaks_i, peaks, peaks_f), axis=0)
        idx_peaks = np.where(peaks.T)[::-1]
        N_peaks = len(idx_peaks[0])
        spec = mag * np.exp(1j * gen.convert(phase, astype=np.ndarray))

        for (f1, t1), (f2, t2) in zip(islice(zip(*idx_peaks), 0, N_peaks - 1),
                                      islice(zip(*idx_peaks), 1, None)):
            if t1 != t2:
                f1_to_end = np.arange(f1 + 1, bpd.shape[0])
                new_ph = np.angle(
                    spec[f1, t1]
                    * cls._win_spec_fft[f1_to_end - f1]  # / cls._win_spec_fft[0]
                )
                if pkg == np:
                    phase[f1_to_end, t1] = new_ph
                else:
                    phase[f1_to_end, t1] = torch.from_numpy(new_ph).to(phase)

                zero_to_f2 = np.arange(0, f2)
                new_ph = np.angle(
                    spec[f2, t2]
                    * cls._win_spec_fft[zero_to_f2 - f2]  # / cls._win_spec_fft[0]
                )
                if pkg == np:
                    phase[zero_to_f2, t2] = new_ph
                else:
                    phase[zero_to_f2, t2] = torch.from_numpy(new_ph).to(phase)
            else:
                t = t1

                f1to2 = np.arange(f1 + 1, f2)
                new_ph = np.angle(
                    (spec[f1, t] * cls._win_spec_fft[f1to2 - f1]
                     + spec[f2, t] * cls._win_spec_fft[f1to2 - f2])
                    # / cls._win_spec_fft[0]
                )
                if pkg == np:
                    phase[f1to2, t] = new_ph
                else:
                    phase[f1to2, t] = torch.from_numpy(new_ph).to(phase)

        return gen.expand_dims(phase, -1)  # F, T, 1


# indexing 다시 해야함
# def complex2magphase(a: ndarray, concat=True) \
#         -> Union[ndarray, Sequence[ndarray]]:
#     a_mag = np.abs(a[..., -1])
#     a_phase = np.angle(a[..., -1])
#     if concat:
#         return np.cat((a[..., :-1].real, a_mag, a_phase), axis=-1)
#     else:
#         return np.cat((a[..., :-1].real, a_mag), axis=-1), a_phase


# def realimag2complex(a: ndarray) -> ndarray:
#     a_complex = a[..., -2] + 1j * a[..., -1]
#
#     return np.cat((a[..., :-2], a_complex), axis=-1)


class ITransformer(metaclass=ABCMeta):
    no_norm_channels = []

    @staticmethod
    @abstractmethod
    def use_phase() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        pass

    @staticmethod
    @abstractmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        pass

    @staticmethod
    @abstractmethod
    def inverse(a: TensArr) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def inverse_(a: TensArr) -> tuple:
        pass


class MagTransformer(ITransformer):
    @staticmethod
    def use_phase() -> bool:
        return False

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return a

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return a

    @classmethod
    def inverse(cls, a: TensArr) -> tuple:
        return a, None

    @classmethod
    def inverse_(cls, a: TensArr) -> tuple:
        return a, None


class LogMagTransformer(ITransformer):
    @staticmethod
    def use_phase() -> bool:
        return False

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return LogModule.log(a)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return LogModule.log_(a)

    @classmethod
    def inverse(cls, a: TensArr) -> tuple:
        return LogModule.exp(a), None

    @classmethod
    def inverse_(cls, a: TensArr) -> tuple:
        return LogModule.exp_(a), None


class ReImTransformer(ITransformer):
    @staticmethod
    def use_phase() -> bool:
        return True

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return magphase2realimag(a, a_phase)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return magphase2realimag(a, a_phase)

    @classmethod
    def inverse(cls, a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        return b, b_phase

    @classmethod
    def inverse_(cls, a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        return b, b_phase


class LogReImTransformer(ReImTransformer):
    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = LogModule.log(a,
                          # only_I=True
                          )
        return super(LogReImTransformer, LogReImTransformer).transform(a, a_phase)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = LogModule.log_(a,
                           # only_I=True
                           )
        return super(LogReImTransformer, LogReImTransformer).transform_(a, a_phase)

    @classmethod
    def inverse(cls, a: TensArr) -> tuple:
        b, b_phase = super(LogReImTransformer, LogReImTransformer).inverse(a)
        b = LogModule.exp_(b,
                           # only_I=True
                           )
        return b, b_phase

    @classmethod
    def inverse_(cls, a: TensArr) -> tuple:
        b, b_phase = super(LogReImTransformer, LogReImTransformer).inverse_(a)
        b = LogModule.exp_(b,
                           # only_I=True
                           )
        return b, b_phase


class LogMagBPDTransformer(ITransformer):
    no_norm_channels = [-1]

    phase_init: TensArr = None
    x_mag: TensArr = None
    along_freq = True

    @staticmethod
    def use_phase() -> bool:
        return True

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return gen.cat((LogModule.log(a), BPD.phase2bpd(a_phase) / np.pi), axis=-1)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        bpd = BPD.phase2bpd_(a_phase)
        bpd /= np.pi
        return gen.cat((LogModule.log_(a), bpd), axis=-1)

    @classmethod
    def inverse(cls, a: TensArr) -> tuple:
        pkg = gen.dict_package[type(a)]
        mask = a[..., -2:-1] / cls.x_mag
        mask[pkg.isnan(mask)] = 0.
        mask[pkg.isinf(mask)] = 0.
        return (LogModule.exp(a[..., :-1]),
                BPD.bpd2phase_(a[..., -1:] * np.pi,
                               a[..., -2:-1],
                               cls.phase_init, mask, cls.along_freq))

    @classmethod
    def inverse_(cls, a: TensArr) -> tuple:
        pkg = gen.dict_package[type(a)]
        a[..., -1] *= np.pi
        mask = a[..., -2:-1] / cls.x_mag
        mask[pkg.isnan(mask)] = 0.
        mask[pkg.isinf(mask)] = 0.
        return (LogModule.exp_(a[..., :-1]),
                BPD.bpd2phase_(a[..., -1:],
                               a[..., -2:-1],
                               cls.phase_init, mask, cls.along_freq))


class LogModule:
    eps = hp.eps_for_log

    @classmethod
    def log(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **_KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps) / cls.eps)
            b[..., :3] = a[..., :3] * log_r / (r + cls.eps)
            if not only_I:
                b[..., 3] = pkg.log10((a[..., 3] + cls.eps) / cls.eps)
        else:
            if not only_I:
                b = pkg.log10((a + cls.eps) / cls.eps)
        return b

    @classmethod
    def log_(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **_KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.eps) / cls.eps)
            a[..., :3] *= log_r
            a[..., :3] /= (r + cls.eps)
            if not only_I:
                a[..., 3] = pkg.log10((a[..., 3] + cls.eps) / cls.eps)
        else:
            if not only_I:
                pkg.log10((a + cls.eps) / cls.eps, out=a)
        return a

    @classmethod
    def exp(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **_KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps * (10.**r - 1)
            b[..., :3] = a[..., :3] * exp_r / (r + cls.eps)
            if not only_I:
                b[..., 3] = cls.eps * (10.**a[..., 3] - 1)
        else:
            if not only_I:
                b = cls.eps * (10.**a - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **_KWARGS_SUM[pkg])**0.5
            exp_r = cls.eps * (10.**r - 1)
            a[..., :3] *= exp_r
            a[..., :3] /= (r + cls.eps)
            if not only_I:
                a[..., 3] = cls.eps * (10.**a[..., 3] - 1)
        else:
            if not only_I:
                a = cls.eps * (10.**a - 1)
        return a
