from abc import ABCMeta, abstractmethod

import numpy as np
import torch

import config as cfg
import generic as gen
from generic import TensArr, TensArrOrSeq


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
    USE_PHASE: bool

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
    USE_PHASE = False

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return a

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return a

    @staticmethod
    def inverse(a: TensArr) -> tuple:
        return a, None

    @staticmethod
    def inverse_(a: TensArr) -> tuple:
        return a, None


class LogMagTransformer(ITransformer):
    USE_PHASE = False

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return LogModule.log(a)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return LogModule.log_(a)

    @staticmethod
    def inverse(a: TensArr) -> tuple:
        return LogModule.exp(a), None

    @staticmethod
    def inverse_(a: TensArr) -> tuple:
        return LogModule.exp_(a), None


class ReImTransformer(ITransformer):
    USE_PHASE = True

    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return magphase2realimag(a, a_phase)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        return magphase2realimag(a, a_phase)

    @staticmethod
    def inverse(a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        return b, b_phase

    @staticmethod
    def inverse_(a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        return b, b_phase


class LogReImTransformer(ReImTransformer):
    @staticmethod
    def transform(a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = LogModule.log(a,
                          # only_I=True
                          )
        return super().transform(a, a_phase)

    @staticmethod
    def transform_(a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = LogModule.log_(a,
                           # only_I=True
                           )
        return super().transform_(a, a_phase)

    @staticmethod
    def inverse(a: TensArr) -> tuple:
        b, b_phase = super().inverse(a)
        b = LogModule.exp_(b,
                           # only_I=True
                           )
        return b, b_phase

    @staticmethod
    def inverse_(a: TensArr) -> tuple:
        b, b_phase = super().inverse_(a)
        b = LogModule.exp_(b,
                           # only_I=True
                           )
        return b, b_phase


class LogModule:
    KWARGS_SUM = {np: dict(keepdims=True), torch: dict(keepdim=True)}
    EPS = cfg.EPS_FOR_LOG

    @classmethod
    def log(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.EPS) / cls.EPS)
            b[..., :3] = a[..., :3] * log_r / (r + cls.EPS)
            if not only_I:
                b[..., 3] = pkg.log10((a[..., 3] + cls.EPS) / cls.EPS)
        else:
            if not only_I:
                b = pkg.log10((a + cls.EPS) / cls.EPS)
        return b

    @classmethod
    def log_(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.EPS) / cls.EPS)
            a[..., :3] *= log_r
            a[..., :3] /= (r + cls.EPS)
            if not only_I:
                a[..., 3] = pkg.log10((a[..., 3] + cls.EPS) / cls.EPS)
        else:
            if not only_I:
                pkg.log10((a + cls.EPS) / cls.EPS, out=a)
        return a

    @classmethod
    def exp(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            exp_r = cls.EPS * (10.**r - 1)
            b[..., :3] = a[..., :3] * exp_r / (r + cls.EPS)
            if not only_I:
                b[..., 3] = cls.EPS * (10.**a[..., 3] - 1)
        else:
            if not only_I:
                b = cls.EPS * (10.**a - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr, only_I=False) -> TensArr:
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            exp_r = cls.EPS * (10.**r - 1)
            a[..., :3] *= exp_r
            a[..., :3] /= (r + cls.EPS)
            if not only_I:
                a[..., 3] = cls.EPS * (10.**a[..., 3] - 1)
        else:
            if not only_I:
                a = cls.EPS * (10.**a - 1)
        return a
