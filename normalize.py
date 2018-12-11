from abc import ABCMeta, abstractmethod
import multiprocessing as mp
from typing import Tuple, List, Dict, Sequence
from itertools import islice

import deepdish as dd

import numpy as np

import torch

import generic as gen
from generic import TensArr

EPS = 1e-4
USE_ONLY_X = False


class NormalizationBase(dd.util.Saveable, metaclass=ABCMeta):
    __slots__ = 'constnames', 'consts', '__cpu', '__cuda'

    def __init__(self, constnames: Sequence[str], consts: Sequence):
        super().__init__()
        self.constnames = constnames
        if consts[0].dtype == np.float64:
            consts = list([a.astype(np.float32) for a in consts])

        self.consts = consts if type(consts) == list else list(consts)
        self.__cpu = None
        self.__cuda = None

    @staticmethod
    def _tensor(consts, device=torch.device('cpu')):
        return [torch.tensor(a, device=device, dtype=torch.float32) for a in consts]

    @staticmethod
    def _calc_per_file(tup: Tuple) -> Tuple:
        fname, xname, yname, list_func, args_x, args_y = tup
        try:
            data_dict = dd.io.load(fname)
        except:  # noqa: E722
            raise Exception(fname)
        x, y = data_dict[xname], data_dict[yname]
        result_x = {f: f(x, arg) for f, arg in zip(list_func, args_x)}
        result_y = {f: f(y, arg) for f, arg in zip(list_func, args_y)}

        print('.', end='', flush=True)
        return result_x, result_y

    def _get_consts_like(self, a: TensArr, xy: str):
        assert xy == 'x' or xy == 'y'
        shft = int(xy == 'y') if not USE_ONLY_X else 0

        ch = range(3, 4) if a.shape[-1] == 1 else range(0, 4)

        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                if not self.__cpu:
                    self.__cpu = self._tensor(self.consts, device=a.device)
                result = self.__cpu
            else:
                if not self.__cuda:
                    self.__cuda = self._tensor(self.consts, device=a.device)
                result = self.__cuda
        else:
            result = self.consts

        return [c[..., ch] if c.shape else c for c in islice(result, shft, None, 2)]

    def save_to_dict(self) -> Dict:
        # return {k: a for k, a in self.__dict__.items()
        #         if not k.startswith('_')}
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
            x = self.consts[2*idx]
            y = self.consts[2*idx + 1]

            result += f'{name}: '
            if hasattr(x, 'shape') and x.shape:
                result += f'{x.shape}, {y.shape}\t'
            else:
                result += f'{x}, {y}\t'

        return result[:-1]

    @abstractmethod
    def normalize(self, a: TensArr, xy: str) -> TensArr:
        pass

    @abstractmethod
    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        pass

    @abstractmethod
    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        pass

    @abstractmethod
    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        pass


class MeanStdNormalization(NormalizationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def sum_(cls, a: np.ndarray, *args) -> np.ndarray:
        return a.sum(axis=1, keepdims=True)

    @classmethod
    def sq_dev(cls, a: np.ndarray, mean_a) -> np.ndarray:
        return ((a - mean_a)**2).sum(axis=1, keepdims=True)

    @classmethod
    def create(cls, all_files: List[str], xname: str, yname: str, *, need_mean=True):
        # Calculate summation & size (parallel)
        list_fn = (np.size, cls.sum_) if need_mean else (np.size,)
        args_none = (None,)*len(list_fn)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(
            cls._calc_per_file,
            [(fname, xname, yname, list_fn, args_none, args_none)
             for fname in all_files]
        )
        print()

        sum_size_x = np.sum([item[0][np.size] for item in result])
        sum_size_y = np.sum([item[1][np.size] for item in result])
        if need_mean:
            sum_x = np.sum([item[0][cls.sum_] for item in result], axis=0)
            sum_y = np.sum([item[1][cls.sum_] for item in result], axis=0)
            mean_x = sum_x/(sum_size_x//sum_x.size)
            mean_y = sum_y/(sum_size_y//sum_y.size)
            # mean_x = sum_x[..., :3] / (sum_size_x//sum_x[..., :3].size)
            # mean_y = sum_y[..., :3] / (sum_size_y//sum_y[..., :3].size)
        else:
            mean_x = 0.
            mean_y = 0.
        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        result = pool.map(
            cls._calc_per_file,
            [(fname, xname, yname, (cls.sq_dev,), (mean_x,), (mean_y,))
             for fname in all_files]
        )
        pool.close()

        sum_sq_dev_x = np.sum([item[0][cls.sq_dev] for item in result], axis=0)
        sum_sq_dev_y = np.sum([item[1][cls.sq_dev] for item in result], axis=0)

        std_x = np.sqrt(sum_sq_dev_x/(sum_size_x//sum_sq_dev_x.size) + 1e-5)
        std_y = np.sqrt(sum_sq_dev_y/(sum_size_y//sum_sq_dev_y.size) + 1e-5)
        print('Calculated Std')

        return cls(('mean', 'std'), (mean_x, mean_y, std_x, std_y))

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a, xy)

        return (a - mean)/std

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a, xy)
        a -= mean
        a /= std
        return a

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a, xy)

        return a*std + mean

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a, xy)

        a *= std
        a += mean
        return a


class MinMaxNormalization(NormalizationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def min_(cls, a: np.ndarray, *args) -> np.ndarray:
        return a.min(axis=1, keepdims=True)

    @classmethod
    def max_(cls, a: np.ndarray, *args) -> np.ndarray:
        return a.max(axis=1, keepdims=True)

    @classmethod
    def create(cls, all_files: List[str], xname: str, yname: str):
        # Calculate summation & no. of total frames (parallel)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(
            cls._calc_per_file,
            [(fname, xname, yname, (cls.min_, cls.max_), (None,)*2, (None,)*2)
             for fname in all_files]
        )
        pool.close()

        min_x = np.min([res[0][cls.min_] for res in result], axis=0)
        min_y = np.min([res[1][cls.min_] for res in result], axis=0)
        max_x = np.max([res[0][cls.max_] for res in result], axis=0)
        max_y = np.max([res[1][cls.max_] for res in result], axis=0)
        print('Calculated Min/Max.')

        return cls(('min', 'max'), (min_x, min_y, max_x, max_y))

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a, xy)

        return (a - min_)/(max_ - min_) - 0.5

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a, xy)

        a -= min_
        a /= (max_ - min_)
        a -= 0.5
        return a

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a, xy)

        return (a + 0.5)*(max_ - min_) + min_

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a, xy)

        a += 0.5
        a *= (max_ - min_)
        a += min_
        return a


class LogInterface:
    signs = {
        (torch, torch.device('cpu')):
            (torch.tensor((1.,), dtype=torch.float32),
             torch.tensor((-1.,), dtype=torch.float32)),
        np:
            (np.array((1,), dtype=np.int8),
             np.array((-1,), dtype=np.int8)),
    }

    @classmethod
    def sign_like(cls, a: TensArr) -> Tuple[TensArr, TensArr]:
        if type(a) == torch.Tensor:
            if (torch, a.device) not in cls.signs:
                cls.signs[(torch, a.device)] = (
                    torch.tensor((1.,), dtype=torch.float32, device=a.device),
                    torch.tensor((-1.,), dtype=torch.float32, device=a.device)
                )
            return cls.signs[(torch, a.device)]
        else:
            return cls.signs[np]

    @classmethod
    def log(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            b[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                          *pkg.log10((pkg.abs(a[..., :3]) + EPS)/EPS))
            b[..., 3] = pkg.log10((a[..., 3] + EPS)/EPS)
        else:
            b = pkg.log10((a + EPS)/EPS)
        return b

    @classmethod
    def log_(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        if a.shape[-1] == 4:
            a[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                          *pkg.log10((pkg.abs(a[..., :3]) + EPS)/EPS))
            a[..., 3] = pkg.log10((a[..., 3] + EPS)/EPS)
        else:
            a = pkg.log10((a + EPS)/EPS)
        return a

    @classmethod
    def exp(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        b = pkg.empty_like(a)
        if a.shape[-1] == 4:
            b[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                          *EPS*(10.**pkg.abs(a[..., :3]) - 1))
            b[..., 3] = EPS*(10.**a[..., 3] - 1)
        else:
            b = EPS*(10.**a - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        if a.shape[-1] == 4:
            a[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                          *EPS*(10.**pkg.abs(a[..., :3]) - 1))
            a[..., 3] = EPS*(10.**a[..., 3] - 1)
        else:
            a = EPS*(10.**a - 1)
        return a


class LogMeanStdNormalization(MeanStdNormalization, LogInterface):
    @classmethod
    def sum_(cls, a: np.ndarray, *args) -> np.ndarray:
        return super().sum_(cls.log_(a))

    @classmethod
    def sq_dev(cls, a: np.ndarray, mean_a) -> np.ndarray:
        return super().sq_dev(cls.log_(a), mean_a)

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        b = self.log(a)
        return super().normalize_(b, xy)

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        a = self.log_(a)
        return super().normalize_(a, xy)

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        b = super().denormalize(a, xy)
        return self.exp_(b)

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        a = super().denormalize_(a, xy)
        return self.exp_(a)


class LogMinMaxNormalization(MinMaxNormalization, LogInterface):
    @classmethod
    def min_(cls, a: np.ndarray, *args) -> np.ndarray:
        return super().min_(cls.log_(a))

    @classmethod
    def max_(cls, a: np.ndarray, *args) -> np.ndarray:
        return super().max_(cls.log_(a))

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        b = self.log(a)
        return super().normalize_(b, xy)

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        a = self.log_(a)
        return super().normalize_(a, xy)

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        b = super().denormalize(a, xy)
        return self.exp_(b)

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        a = super().denormalize_(a, xy)
        return self.exp_(a)
