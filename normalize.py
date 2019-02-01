import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from itertools import islice
from typing import Dict, List, Sequence, Tuple

import deepdish as dd
import numpy as np
import torch

import config as cfg
import generic as gen
from generic import TensArr, TensArrOrSeq


class NormalizationBase(dd.util.Saveable, metaclass=ABCMeta):
    __slots__ = 'constnames', 'consts', '__cpu', '__cuda'

    def __init__(self, constnames: Sequence[str], consts: Sequence):
        super().__init__()
        self.constnames = constnames
        if consts[0].dtype == np.float64:
            consts = list([a.astype(np.float32) for a in consts])

        self.consts = consts if type(consts) == list else list(consts)
        self.__cpu = None
        self.__cuda = dict()

    @staticmethod
    def _tensor(consts, device=torch.device('cpu')):
        return [torch.tensor(a, device=device, dtype=torch.float32) for a in consts]

    @staticmethod
    def _calc_per_file(tup: Tuple) -> Tuple:
        fname, list_func, args_x, args_y = tup[:4]
        if len(tup) == 5:
            use_phase = tup[4]
        else:
            use_phase = False

        while True:
            try:
                if use_phase:
                    x, x_phase, y, y_phase \
                        = dd.io.load(fname, group=(cfg.IV_DATA_NAME['x'],
                                                   cfg.IV_DATA_NAME['x_phase'],
                                                   cfg.IV_DATA_NAME['y'],
                                                   cfg.IV_DATA_NAME['y_phase'],
                                                   ))
                else:
                    x, y = dd.io.load(fname, group=(cfg.IV_DATA_NAME['x'],
                                                    cfg.IV_DATA_NAME['y'],
                                                    ))
                    x_phase = None
                    y_phase = None
                break
            except:  # noqa: E722
                print(fname)
                fname_sq = fname.replace('IV_sqrt', 'IV')
                iv_dict = dd.io.load(fname_sq)
                iv_dict['IV_free'][..., -1] = np.sqrt(iv_dict['IV_free'][..., -1])
                iv_dict['IV_room'][..., -1] = np.sqrt(iv_dict['IV_room'][..., -1])
                dd.io.save(fname, iv_dict, compression=None)

        result_x = {f: f(x, x_phase, arg) for f, arg in zip(list_func, args_x)}
        result_y = {f: f(y, y_phase, arg) for f, arg in zip(list_func, args_y)}

        print('.', end='', flush=True)
        return result_x, result_y

    def _get_consts_like(self, xy: str, a: TensArr):
        assert xy == 'x' or xy == 'y'
        shft = int(xy == 'y') if not cfg.NORM_USING_ONLY_X else 0

        ch = slice(-a.shape[-1], None)

        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                if not self.__cpu:
                    self.__cpu = self._tensor(self.consts, device=a.device)
                result = self.__cpu
            else:
                if a.device not in self.__cuda:
                    self.__cuda[a.device] = self._tensor(self.consts, device=a.device)
                result = self.__cuda[a.device]
        else:
            result = self.consts

        return [c[..., ch] if c.shape else c for c in islice(result, shft, None, 2)]

    def save_to_dict(self, only_consts=False) -> Dict:
        # return {k: a for k, a in self.__dict__.items()
        #         if not k.startswith('_')}
        if only_consts:
            return {f'const_{idx}': const.squeeze()
                    for idx, const in enumerate(self.consts)}
        else:
            return {s: getattr(self, s) for s in self.__slots__
                    if hasattr(self, s) and not s.startswith('_')}

    @classmethod
    def load_from_dict(cls, d: Dict):
        return cls(**d)

    def __len__(self):
        return len(self.consts)

    def __str__(self):
        result = ''
        for idx, name in enumerate(self.constnames):
            x = self.consts[2 * idx]
            y = self.consts[2 * idx + 1]

            result += f'{name}: '
            if hasattr(x, 'shape') and x.shape:
                result += f'{x.shape}, {y.shape}\t'
            else:
                result += f'{x}, {y}\t'

        return result[:-1]

    @abstractmethod
    def normalize(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
        pass

    @abstractmethod
    def normalize_(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
        pass

    @abstractmethod
    def denormalize(self, xy: str, a: TensArr) -> TensArrOrSeq:
        pass

    @abstractmethod
    def denormalize_(self, xy: str, a: TensArr) -> TensArrOrSeq:
        pass


class MeanStdNormalization(NormalizationBase):
    USE_PHASE = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def pre(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        return a

    @classmethod
    def pre_(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        return a

    @classmethod
    def post(cls, a: TensArr) -> tuple:
        return a, None

    @classmethod
    def post_(cls, a: TensArr) -> tuple:
        return a, None

    @classmethod
    def _size(cls, a: np.ndarray, a_phase: np.ndarray, *args) -> int:
        return np.size(cls.pre_(a, a_phase))

    @classmethod
    def _sum(cls, a: np.ndarray, a_phase: np.ndarray, *args) -> np.ndarray:
        return cls.pre_(a, a_phase).sum(axis=1, keepdims=True)

    @classmethod
    def _sq_dev(cls, a: np.ndarray, a_phase: np.ndarray, *args) -> np.ndarray:
        mean_a = args[0]
        return ((cls.pre_(a, a_phase) - mean_a)**2).sum(axis=1, keepdims=True)

    @classmethod
    def create(cls, all_files: List[str], *, need_mean=True):
        # Calculate summation & size (parallel)
        list_fn = (cls._size, cls._sum) if need_mean else (cls._size,)
        args_none = (None,) * len(list_fn)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(
            cls._calc_per_file,
            [(fname, list_fn, args_none, args_none, cls.USE_PHASE)
             for fname in all_files]
        )
        print()

        sum_size_x = np.sum([item[0][cls._size] for item in result])
        sum_size_y = np.sum([item[1][cls._size] for item in result])
        if need_mean:
            sum_x = np.sum([item[0][cls._sum] for item in result], axis=0)
            sum_y = np.sum([item[1][cls._sum] for item in result], axis=0)
            mean_x = sum_x / (sum_size_x // sum_x.size)
            mean_y = sum_y / (sum_size_y // sum_y.size)
            # mean_x = sum_x[..., :3] / (sum_size_x//sum_x[..., :3].size)
            # mean_y = sum_y[..., :3] / (sum_size_y//sum_y[..., :3].size)
        else:
            mean_x = 0.
            mean_y = 0.
        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        result = pool.map(
            cls._calc_per_file,
            [(fname, (cls._sq_dev,), (mean_x,), (mean_y,), cls.USE_PHASE)
             for fname in all_files]
        )
        pool.close()

        sum_sq_dev_x = np.sum([item[0][cls._sq_dev] for item in result], axis=0)
        sum_sq_dev_y = np.sum([item[1][cls._sq_dev] for item in result], axis=0)

        std_x = np.sqrt(sum_sq_dev_x / (sum_size_x // sum_sq_dev_x.size) + 1e-5)
        std_y = np.sqrt(sum_sq_dev_y / (sum_size_y // sum_sq_dev_y.size) + 1e-5)
        print('Calculated Std')

        return cls(('mean', 'std'), (mean_x, mean_y, std_x, std_y))

    def normalize(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
        b = self.pre(a, a_phase)
        mean, std = self._get_consts_like(xy, b)

        return (b - mean) / (2 * std)

    def normalize_(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = self.pre_(a, a_phase)
        mean, std = self._get_consts_like(xy, a)

        a -= mean
        a /= (2 * std)

        return a

    def denormalize(self, xy: str, a: TensArr) -> TensArrOrSeq:
        mean, std = self._get_consts_like(xy, a)
        tup = self.post_(a * (2 * std) + mean)
        if tup[1] is None:
            return tup[0]
        else:
            return tup

    def denormalize_(self, xy: str, a: TensArr) -> TensArrOrSeq:
        mean, std = self._get_consts_like(xy, a)
        a *= (2 * std)
        a += mean
        tup = self.post_(a)
        if tup[1] is None:
            return tup[0]
        else:
            return tup


# class MinMaxNormalization(NormalizationBase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     @classmethod
#     def _min(cls, a: np.ndarray, *args) -> np.ndarray:
#         return a.min(axis=1, keepdims=True)
#
#     @classmethod
#     def _max(cls, a: np.ndarray, *args) -> np.ndarray:
#         return a.max(axis=1, keepdims=True)
#
#     @classmethod
#     def create(cls, all_files: List[str]):
#         # Calculate summation & no. of total frames (parallel)
#         pool = mp.Pool(mp.cpu_count())
#         result = pool.map(
#             cls._calc_per_file,
#             [(fname, (cls._min, cls._max), (None,) * 2, (None,) * 2)
#              for fname in all_files]
#         )
#         pool.close()
#
#         min_x = np.min([res[0][cls._min] for res in result], axis=0)
#         min_y = np.min([res[1][cls._min] for res in result], axis=0)
#         max_x = np.max([res[0][cls._max] for res in result], axis=0)
#         max_y = np.max([res[1][cls._max] for res in result], axis=0)
#         print('Calculated Min/Max.')
#
#         return cls(('min', 'max'), (min_x, min_y, max_x, max_y))
#
#     def normalize(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         min_, max_ = self._get_consts_like(xy, a)
#
#         return (a - min_) / (max_ - min_) - 0.5
#
#     def normalize_(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         min_, max_ = self._get_consts_like(xy, a)
#
#         a -= min_
#         a /= (max_ - min_)
#         a -= 0.5
#         return a
#
#     def denormalize(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         min_, max_ = self._get_consts_like(xy, a)
#
#         return (a + 0.5) * (max_ - min_) + min_
#
#     def denormalize_(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         min_, max_ = self._get_consts_like(xy, a)
#
#         a += 0.5
#         a *= (max_ - min_)
#         a += min_
#         return a


class LogInterface:
    KWARGS_SUM = {np: dict(keepdims=True), torch: dict(keepdim=True)}
    EPS = cfg.EPS_FOR_LOG

    @classmethod
    def log(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.EPS) / cls.EPS)
            b[..., :3] = a[..., :3] * log_r / (r + cls.EPS)

            b[..., 3] = pkg.log10((a[..., 3] + cls.EPS) / cls.EPS)
        else:
            b = pkg.log10((a + cls.EPS) / cls.EPS)
        return b

    @classmethod
    def log_(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            log_r = pkg.log10((r + cls.EPS) / cls.EPS)
            a[..., :3] *= log_r
            a[..., :3] /= (r + cls.EPS)

            a[..., 3] = pkg.log10((a[..., 3] + cls.EPS) / cls.EPS)
        else:
            pkg.log10((a + cls.EPS) / cls.EPS, out=a)
        return a

    @classmethod
    def exp(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            exp_r = cls.EPS * (10.**r - 1)
            b[..., :3] = a[..., :3] * exp_r / (r + cls.EPS)

            b[..., 3] = cls.EPS * (10.**a[..., 3] - 1)
        else:
            b = cls.EPS * (10.**a - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]

        if a.shape[-1] == 4:
            r = (a[..., :3]**2).sum(-1, **cls.KWARGS_SUM[pkg])**0.5
            exp_r = cls.EPS * (10.**r - 1)
            a[..., :3] *= exp_r
            a[..., :3] /= (r + cls.EPS)

            a[..., 3] = cls.EPS * (10.**a[..., 3] - 1)
        else:
            a = cls.EPS * (10.**a - 1)
        return a


class LogMeanStdNormalization(MeanStdNormalization, LogInterface):
    @classmethod
    def pre(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        return cls.log(a)

    @classmethod
    def pre_(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        return cls.log_(a)

    @classmethod
    def post(cls, a: TensArr) -> tuple:
        return cls.exp(a), None

    @classmethod
    def post_(cls, a: TensArr) -> tuple:
        return cls.exp_(a), None


# class LogMinMaxNormalization(MinMaxNormalization, LogInterface):
#     @classmethod
#     def _min(cls, a: np.ndarray, *args) -> np.ndarray:
#         return super()._min(cls.log_(a))
#
#     @classmethod
#     def _max(cls, a: np.ndarray, *args) -> np.ndarray:
#         return super()._max(cls.log_(a))
#
#     def normalize(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         b = self.log(a)
#         return super().normalize_(xy, b)
#
#     def normalize_(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         a = self.log_(a)
#         return super().normalize_(xy, a)
#
#     def denormalize(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         b = super().denormalize(xy, a)
#         return self.exp_(b)
#
#     def denormalize_(self, xy: str, a: TensArr, a_phase: TensArr = None) -> TensArr:
#         a = super().denormalize_(xy, a)
#         return self.exp_(a)


class ReImMeanStdNormalization(MeanStdNormalization):
    USE_PHASE = True

    @classmethod
    def pre(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        # return magphase2realimag(cls.log(a), a_phase)
        return magphase2realimag(a, a_phase)

    @classmethod
    def pre_(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        # return magphase2realimag(cls.log_(a), a_phase)
        return magphase2realimag(a, a_phase)

    @classmethod
    def post(cls, a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        # return cls.exp_(b), b_phase
        return b, b_phase

    @classmethod
    def post_(cls, a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        # return cls.exp_(b), b_phase
        return b, b_phase

    @classmethod
    def _size(cls, a: np.ndarray, a_phase: np.ndarray, *args) -> int:
        return np.size(a) + np.size(a_phase)


class LogReImMeanStdNormalization(ReImMeanStdNormalization, LogInterface):
    @classmethod
    def pre(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = cls.log(a)
        return magphase2realimag(a, a_phase)

    @classmethod
    def pre_(cls, a: TensArr, a_phase: TensArr = None) -> TensArr:
        a = cls.log_(a)
        return magphase2realimag(a, a_phase)

    @classmethod
    def post(cls, a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        b = cls.exp_(b)
        return b, b_phase

    @classmethod
    def post_(cls, a: TensArr) -> tuple:
        b, b_phase = realimag2magphase(a, concat=False)
        b = cls.exp_(b)
        return b, b_phase


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
# def complex2magphase(a: np.ndarray, concat=True) \
#         -> Union[np.ndarray, Sequence[np.ndarray]]:
#     a_mag = np.abs(a[..., -1])
#     a_phase = np.angle(a[..., -1])
#     if concat:
#         return np.cat((a[..., :-1].real, a_mag, a_phase), axis=-1)
#     else:
#         return np.cat((a[..., :-1].real, a_mag), axis=-1), a_phase


# def realimag2complex(a: np.ndarray) -> np.ndarray:
#     a_complex = a[..., -2] + 1j * a[..., -1]
#
#     return np.cat((a[..., :-2], a_complex), axis=-1)
