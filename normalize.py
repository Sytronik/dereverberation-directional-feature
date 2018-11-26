from abc import ABCMeta, abstractmethod
import multiprocessing as mp
from typing import Tuple, List, Dict

import deepdish as dd

import numpy as np

import torch

import generic as gen
from generic import TensArr


class NormalizationBase(dd.util.Saveable, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__()
        for k, a in kwargs.items():
            self.__dict__[k] = a

        self._LEN = len(kwargs)

        self.__cpu = None
        self.__cuda = None

    def _convert(self, astype: type, device=torch.device('cpu')):
        if astype == torch.Tensor:
            members = {k: gen.convert(a, astype).to(device=device)
                       for k, a in self.save_to_dict().items()}
        else:
            members = {k: gen.convert(a, astype)
                       for k, a in self.save_to_dict().items()}
        return type(self)(**members)

    @staticmethod
    def calc_per_file(tup: Tuple) -> Tuple:
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

    def _get_consts_like(self, a: TensArr):
        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                if not self.__cpu:
                    self.__cpu = self._convert(type(a), device=a.device)
                return self.__cpu
            else:
                if not self.__cuda:
                    self.__cuda = self._convert(type(a), device=a.device)
                return self.__cuda
        else:
            return self

    def save_to_dict(self) -> Dict:
        return {k: a for k, a in self.__dict__.items()
                if not k.startswith('_')}

    @classmethod
    def load_from_dict(cls, d: Dict):
        return cls(**d)

    def __len__(self):
        return self._LEN

    def __str__(self):
        result = ''
        visited = {str}
        for k in self.__dict__:
            name = k[:-2]
            if k.startswith('_') or name in visited:
                continue

            result += f'{name}: '
            x = self.__dict__[f'{name}_x']
            y = self.__dict__[f'{name}_y']
            if hasattr(x, 'shape') and x.shape:
                result += f'{x.shape}, {y.shape}\t'
            else:
                result += f'{x}, {y}\t'
            visited.add(name)

        if result.endswith('\t'):
            result = result[:-1]
        return result

    @abstractmethod
    def __getitem__(self, xy: str) -> Tuple:
        pass

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
    def __init__(self, mean_x, mean_y, std_x, std_y):
        super().__init__(
            mean_x=mean_x,
            mean_y=mean_y,
            std_x=std_x,
            std_y=std_y
        )

    @classmethod
    def sum_(cls, a: np.ndarray, *args) -> np.ndarray:
        return a.sum(axis=1, keepdims=True)

    @classmethod
    def sq_dev(cls, a: np.ndarray, mean_a)-> np.ndarray:
        return ((a - mean_a)**2).sum(axis=1, keepdims=True)

    @classmethod
    def create(cls, all_files: List[str], xname: str, yname: str):
        # Calculate summation & size (parallel)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(
            cls.calc_per_file,
            [(fname, xname, yname, (np.size,), (None,), (None,))
             for fname in all_files]
        )
        print()

        # sum_x = np.sum([item[0]['sum'] for item in result], axis=0)
        # sum_y = np.sum([item[1]['sum'] for item in result], axis=0)
        sum_size_x = np.sum([item[0][np.size] for item in result])
        sum_size_y = np.sum([item[1][np.size] for item in result])

        # mean_x = sum_x[..., :3]/(sum_size_x // sum_x[..., :3].size)
        # mean_y = sum_y[..., :3]/(sum_size_y // sum_y[..., :3].size)
        mean_x = 0.
        mean_y = 0.
        print('Calculated Mean')

        # Calculate squared deviation (parallel)
        result = pool.map(
            cls.calc_sq_dev,
            [(fname, xname, yname, (cls.sq_dev_log,), (mean_x,), (mean_y,))
             for fname in all_files]
        )
        pool.close()

        sum_sq_dev_x = np.sum([item[0][cls.sq_dev] for item in result], axis=0)
        sum_sq_dev_y = np.sum([item[1][cls.sq_dev] for item in result], axis=0)

        std_x = np.sqrt(sum_sq_dev_x/(sum_size_x//sum_sq_dev_x.size) + 1e-5)
        std_y = np.sqrt(sum_sq_dev_y/(sum_size_y//sum_sq_dev_y.size) + 1e-5)
        print('Calculated Std')

        return cls(mean_x, mean_y, std_x, std_y)

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a)[xy]
        b = gen.copy(a)
        # b[..., :3] -= mean
        b /= std
        return (a - mean)/std

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a)[xy]
        # a[..., :3] -= mean
        a -= mean
        a /= std
        return a

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a)[xy]
        b = a*std
        # b[..., :3] += mean
        b += mean
        return b

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        mean, std = self._get_consts_like(a)[xy]
        a *= std
        # a[..., :3] += mean
        a += mean
        return a

    def __getitem__(self, xy: str) -> Tuple:
        try:
            mean = self.__dict__[f'mean_x']
            std = self.__dict__[f'std_x']
        except AttributeError:
            raise Exception('the index should be "x" or "y"')

        return mean, std


class MinMaxNormalization(NormalizationBase):
    def __init__(self, min_x, min_y, max_x, max_y):
        super().__init__(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y
        )

    @classmethod
    def create(cls, all_files: List[str], xname: str, yname: str):
        # Calculate summation & no. of total frames (parallel)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(
            cls.calc_minmax,
            [(fname, xname, yname, (np.min, np.max), (None,)*2, (None,)*2)
             for fname in all_files]
        )
        pool.close()

        min_x = np.min([res[0][np.min] for res in result])
        min_y = np.min([res[1][np.min] for res in result])
        max_x = np.max([res[0][np.max] for res in result])
        max_y = np.max([res[1][np.max] for res in result])

        return cls(min_x, min_y, max_x, max_y)

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a)[xy]
        # return (a - mean)/std - min_
        return (a - min_)/(max_ - min_)

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a)[xy]
        # return (a - mean)/std - min_
        a -= min_
        a /= (max_ - min_)
        return a

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a)[xy]
        # return (a + min_)*std + mean
        return a*(max_ - min_) + min_

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        min_, max_ = self._get_consts_like(a)[xy]
        # a += min_
        a *= (max_ - min_)
        a += min_
        return a

    def __getitem__(self, xy: str) -> Tuple:
        try:
            min_ = self.__dict__[f'min_{xy}']
            max_ = self.__dict__[f'max_{xy}']
        except AttributeError:
            raise Exception('the index should be "x" or "y"')

        return min_, max_


class LogMeanStdNormalization(MeanStdNormalization):
    EPS = 1
    signs = {
        (torch, torch.device('cpu')):
            (torch.tensor((1.,), dtype=torch.float32),
             torch.tensor((-1.,), dtype=torch.float32)),
        np:
            (np.array((1,), dtype=np.int8),
             np.array((-1,), dtype=np.int8)),
    }

    @classmethod
    def sign_like(cls, a: TensArr) -> TensArr:
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

        b = pkg.zeros_like(a)
        b[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                      * pkg.log10((pkg.abs(a[..., :3]) + cls.EPS)/cls.EPS))
        b[..., 3] = pkg.log10((a[..., 3] + cls.EPS)/cls.EPS)
        return b

    @classmethod
    def log_(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        a[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                      * pkg.log10((pkg.abs(a[..., :3]) + cls.EPS)/cls.EPS))
        a[..., 3] = pkg.log10((a[..., 3] + cls.EPS)/cls.EPS)
        return a

    @classmethod
    def exp(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        b = pkg.zeros_like(a)
        b[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                      * cls.EPS*(10.**pkg.abs(a[..., :3]) - 1))
        b[..., 3] = cls.EPS*(10.**a[..., 3] - 1)
        return b

    @classmethod
    def exp_(cls, a: TensArr) -> TensArr:
        pkg = gen.dict_package[type(a)]
        plus, minus = cls.sign_like(a)

        a[..., :3] = (pkg.where(a[..., :3] > 0, plus, minus)
                      * cls.EPS*(10.**pkg.abs(a[..., :3]) - 1))
        a[..., 3] = cls.EPS*(10.**a[..., 3] - 1)
        return a

    @classmethod
    def sum_(cls, a: np.ndarray, *args) -> np.ndarray:
        return super().sum_(cls.log_(a))

    @classmethod
    def sq_dev(cls, a: np.ndarray, mean_a) -> np.ndarray:
        return super().sum_(cls.log_(a), mean_a)

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
