import pdb  # noqa: F401

import numpy as np
import deepdish as dd

import torch
from torch.utils.data import Dataset

from typing import Tuple, Union, List, Dict
from abc import ABCMeta, abstractmethod

import os
from os import path
from copy import copy
import multiprocessing as mp

import generic as gen
import mypath


class NormalizationBase(dd.util.Saveable, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(NormalizationBase, self).__init__()
        for k, a in kwargs.items():
            self.__dict__[k] = a

        self._LEN = len(kwargs)

        self._cpu = None
        self._cuda = None

    def _convert(self, astype: type, device=torch.device('cpu')):
        if astype == torch.Tensor:
            members = {k: gen.convert(a, astype).to(device=device)
                       for k, a in self.save_to_dict().items()}
        else:
            members = {k: gen.convert(a, astype)
                       for k, a in self.save_to_dict().items()}
        return type(self)(**members)

    def _get_as_type_of(self, a: gen.TensArr):
        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                if not self._cpu:
                    self._cpu = self._convert(type(a), device=a.device)
                return self._cpu
            else:
                if not self._cuda:
                    self._cuda = self._convert(type(a), device=a.device)
                return self._cuda
        else:
            return self

    def save_to_dict(self):
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
            core = k[:-2]
            if k.startswith('_') or core in visited:
                continue

            result += f'{core}: '
            x = self.__dict__[core + "_x"]
            y = self.__dict__[core + "_y"]
            if hasattr(x, 'shape') and x.shape:
                result += (f'{x.shape}, {y.shape}\n')
            else:
                result += (f'{x}, {y}\n')
            visited.add(core)

        if result.endswith('\n'):
            result = result[:-1]
        return result

    @abstractmethod
    def __getitem__(self, xy: str) -> Tuple:
        pass

    @abstractmethod
    def normalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        pass

    @abstractmethod
    def normalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        pass

    @abstractmethod
    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        pass

    @abstractmethod
    def denormalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        pass

    @abstractmethod
    def _get_consts_as_type_of(self, a: gen.TensArr, xy: str) -> Tuple:
        pass


class MeanStdNormalization(NormalizationBase):
    def __init__(self, mean_x, mean_y, std_x, std_y):
        super(MeanStdNormalization, self).__init__(
            mean_x=mean_x,
            mean_y=mean_y,
            std_x=std_x,
            std_y=std_y
        )

    @classmethod
    def create(cls, all_files: List[str], XNAME: str, YNAME: str):
        # Calculate summation & no. of total frames (parallel)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(cls.calc_size_sum,
                          [(fname, XNAME, YNAME)
                           for fname in all_files])

        # sum_x = np.sum([item[0]['sum'] for item in result])
        # sum_y = np.sum([item[1]['sum'] for item in result])
        sum_size_x = np.sum([item[0]['size'] for item in result])
        sum_size_y = np.sum([item[1]['size'] for item in result])

        mean_x = 0.  # sum_x / sum_size_x
        mean_y = 0.  # sum_y / sum_size_y

        result = pool.map(cls.calc_sq_dev,
                          [(fname, XNAME, YNAME, mean_x, mean_y)
                           for fname in all_files])
        pool.close()

        sum_sq_dev_x = np.sum([item[0] for item in result], axis=0)
        sum_sq_dev_y = np.sum([item[1] for item in result], axis=0)
        sum_size_x //= sum_sq_dev_x.size
        sum_size_y //= sum_sq_dev_y.size
        std_x = np.sqrt(sum_sq_dev_x / sum_size_x + 1e-5)
        std_y = np.sqrt(sum_sq_dev_y / sum_size_y + 1e-5)

        return cls(mean_x, mean_y, std_x, std_y)

    @staticmethod
    def calc_size_sum(tup: Tuple[str, str, str]) -> Tuple[Dict, Dict]:
        """
        (fname, XNAME, YNAME) -> (sum of x, length of x, sum of y, length of y)
        """
        fname, XNAME, YNAME = tup

        try:
            data_dict = dd.io.load(fname)
        except:  # noqa: E722
            raise Exception(fname)
        x, y = data_dict[XNAME], data_dict[YNAME]

        ss_x = {'size': x.size,
                # 'sum': x.sum(),
                }
        ss_y = {'size': y.size,
                # 'sum': y.sum(),
                }

        print('.', end='', flush=True)
        return (ss_x, ss_y)

    @staticmethod
    def calc_sq_dev(tup: Tuple) -> Tuple[Dict, Dict]:
        """
        (fname, XNAME, YNAME, mean of x, mean of y)
        -> (sum of deviation of x, sum of deviation of y)
        """
        fname, XNAME, YNAME, mean_x, mean_y = tup

        data_dict = dd.io.load(fname)
        x, y = data_dict[XNAME], data_dict[YNAME]

        sq_dev_x = ((x - mean_x)**2).sum(axis=(1, 2), keepdims=True)
        sq_dev_y = ((y - mean_y)**2).sum(axis=(1, 2), keepdims=True)

        print('.', end='', flush=True)
        return (sq_dev_x, sq_dev_y)

    def normalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std = self._get_consts_as_type_of(a, xy)
        return (a - mean) / std

    def normalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std = self._get_consts_as_type_of(a, xy)
        a -= mean
        a /= std
        return a

    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std = self._get_consts_as_type_of(a, xy)
        return a * std + mean

    def denormalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std = self._get_consts_as_type_of(a, xy)
        a *= std
        a += mean
        return a

    def _get_consts_as_type_of(self, a: gen.TensArr, xy: str) -> Tuple:
        mean, std = self._get_as_type_of(a)[xy]

        if hasattr(a, 'shape') and std.shape[0] == a.shape[-2]:
            std = gen.transpose(std, (2, 0, 1))
        return mean, std

    def __getitem__(self, xy: str) -> Tuple:
        try:
            mean = self.__dict__[f'mean_{xy}']
            std = self.__dict__[f'std_{xy}']
        except AttributeError:
            raise Exception('the index should be "x" or "y"')

        return mean, std


class MinMaxNormalization(NormalizationBase):
    def __init__(self, min_x, min_y, max_x, max_y):
        super(MeanStdNormalization, self).__init__(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y
        )

    @classmethod
    def create(cls, all_files: List[str], XNAME: str, YNAME: str):
        # Calculate summation & no. of total frames (parallel)
        pool = mp.Pool(mp.cpu_count())
        result = pool.map(cls.calc_minmax,
                          [(fname, XNAME, YNAME)
                           for fname in all_files])
        pool.close()

        min_x = np.min([res[0]['min'] for res in result])
        min_y = np.min([res[1]['min'] for res in result])
        max_x = np.max([res[0]['max'] for res in result])
        max_y = np.max([res[1]['max'] for res in result])

        return cls(min_x, min_y, max_x, max_y)

    @staticmethod
    def calc_minmax(tup: Tuple[str, str, str]) -> Tuple[Tuple, Tuple]:
        """
        (fname, XNAME, YNAME)->(sum of x, length of x, sum of y, length of y)
        """
        fname, XNAME, YNAME = tup

        try:
            data_dict = dd.io.load(fname)
        except:  # noqa: E722
            raise Exception(fname)

        x, y = data_dict[XNAME], data_dict[YNAME]
        mm_x = {'min': x.min(), 'max': x.max()}
        mm_y = {'min': y.min(), 'max': y.max()}
        print('.', end='', flush=True)
        return (mm_x, mm_y)

    def normalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        min_, max_ = self._get_consts_as_type_of(a, xy)
        # return (a - mean) / std - min_
        return (a - min_) / (max_ - min_)

    def normalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        min_, max_ = self._get_consts_as_type_of(a, xy)
        # return (a - mean) / std - min_
        a -= min_
        a /= (max_ - min_)
        return a

    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        min_, max_ = self._get_consts_as_type_of(a, xy)
        # return (a + min_) * std + mean
        return a * (max_ - min_) + min_

    def denormalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        min_, max_ = self._get_consts_as_type_of(a, xy)
        # a += min_
        a *= (max_ - min_)
        a += min_
        return a

    def _get_consts_as_type_of(self, a: gen.TensArr, xy: str) -> Tuple:
        min_, max_ = self._get_as_type_of(a)[xy]

        return min_, max_

    def __getitem__(self, xy: str) -> Tuple:
        try:
            min_ = self.__dict__[f'min_{xy}']
            max_ = self.__dict__[f'max_{xy}']
        except AttributeError:
            raise Exception('the index should be "x" or "y"')

        return min_, max_


class IVDataset(Dataset):
    """
    <Instance Variable>
    (not splitted)
    PATH
    XNAME
    YNAME
    normconst

    (to be splitted)
    _all_files
    """

    def __init__(self, kind_data: str, XNAME: str, YNAME: str,
                 N_file=-1, doNormalize=True):
        self.PATH = mypath.path(f'iv_{kind_data}')
        self.XNAME = XNAME
        self.YNAME = YNAME

        # fname_list: The name of the file
        # that has information about data file list, mean, std, ...
        fname_list = path.join(self.PATH, f'list_files_{N_file}_no_cut.h5')

        if path.isfile(fname_list):
            self._all_files, normconst = dd.io.load(fname_list)
            if doNormalize:
                self._normconst_np \
                    = MeanStdNormalization.load_from_dict(normconst)
                print(self._normconst_np)
            else:
                self._normconst_np = None
                print()
        else:
            # search all data files
            _all_files = [f.path
                          for f in os.scandir(self.PATH)
                          if f.name.endswith('.h5')
                          and f.name != 'metadata.h5'
                          and not f.name.startswith('list_files_')]
            self._all_files = np.random.permutation(_all_files)
            if N_file != -1:
                self._all_files = self._all_files[:N_file]

            if doNormalize:
                self._normconst_np \
                    = MeanStdNormalization.create(self._all_files,
                                                  XNAME, YNAME)
            else:
                self._normconst_np = None

            dd.io.save(fname_list,
                       (self._all_files,
                        self._normconst_np.save_to_dict()
                        if self._normconst_np else None,
                        )
                       )
        print(f'{N_file} files prepared from {path.basename(self.PATH)}.')

    def __len__(self):
        return len(self._all_files)

    def __getitem__(self, idx: int):
        # File Open (with Slicing)
        data_dict = dd.io.load(self._all_files[idx])
        x = data_dict[self.XNAME]
        y = data_dict[self.YNAME]

        # Normalize
        if self._normconst_np:
            # x = (x - self._normconst_np.min_x) \
            #     / (self._normconst_np.max_x - self._normconst_np.min_x)
            # y = (y - self._normconst_np.min_y) \
            #     / (self._normconst_np.max_y - self._normconst_np.min_y)
            x = self._normconst_np.normalize(x, 'x')
            y = self._normconst_np.normalize(y, 'y')

        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y).permute(2, 0, 1)
        # y = F.pad(y, (0, x.size()[-1]-y.sisz()[-1]))

        sample = {'x': x, 'y': y,
                  # 'fname': self._all_files[idx],
                  }

        return sample

    def pad_collate(self, batch: List) -> Dict:
        max_N_frame = 0
        N_frames_x = [int] * len(batch)
        N_frames_y = [int] * len(batch)
        for idx, item in enumerate(batch):
            N_frames_x[idx] = item['x'].size()[-1]
            N_frames_y[idx] = item['y'].size()[-1]

            if max_N_frame < N_frames_x[idx]:
                max_N_frame = N_frames_x[idx]

        if max_N_frame % 2 == 1:
            max_N_frame += 1

        x = [None] * len(batch)
        y = [None] * len(batch)

        pad_x = np.zeros((*batch[0]['x'].shape[0:2], 1))
        pad_y = np.zeros((*batch[0]['y'].shape[0:2], 1))
        if self._normconst_np:
            pad_x = self._normconst_np.normalize_(pad_x, 'x')
            pad_y = self._normconst_np.normalize_(pad_y, 'y')
            pad_x = pad_x.astype(np.float32, copy=False)
            pad_y = pad_y.astype(np.float32, copy=False)

        with torch.no_grad():
            pad_x = torch.from_numpy(pad_x)
            pad_y = torch.from_numpy(pad_y)

            for idx, item in enumerate(batch):
                x[idx] = torch.cat(
                    (item['x'],) + int(max_N_frame -
                                       N_frames_x[idx]) * (pad_x,),
                    dim=-1
                )
                y[idx] = torch.cat(
                    (item['y'],) + int(max_N_frame -
                                       N_frames_y[idx]) * (pad_y,),
                    dim=-1
                )

        # fnames = [item['fname'] for item in batch]
        return {'x': torch.stack(x), 'y': torch.stack(y),
                'N_frames_x': N_frames_x, 'N_frames_y': N_frames_y,
                # 'fnames': fnames,
                }

    def normalizeOnLike(self, other):
        self._normconst_np = other._normconst_np

    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        return self._normconst_np.denormalize(a, xy)

    def denormalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        return self._normconst_np.denormalize_(a, xy)

    @classmethod
    def split(cls,
              a, ratio: Union[Tuple[float, ...], List[float]]) -> Tuple:
        """
        Split datasets.
        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1
        """
        if type(a) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0
        if mask.sum() > 1:
            raise Exception("Only one element of the parameter 'ratio' "
                            "can have the value of -1")
        if ratio.sum() >= 1:
            raise Exception('The sum of elements of ratio must be 1')
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(a._all_files),
                             dtype=int)
        result = [copy(a) for ii in range(n_split)]
        # all_f_per = np.random.permutation(a._all_files)

        for ii in range(n_split):
            result[ii]._all_files \
                = a._all_files[idx_data[ii]:idx_data[ii + 1]]

        return tuple(result)


def norm_iv(data: gen.TensArr, keep_freq_axis=False,
            parts: Union[str, List[str], Tuple[str]]='all') -> gen.TensArr:
    dim = gen.ndim(data)
    if dim != 3 and dim != 4:
        raise f'Dimension Mismatch: {dim}'

    DICT_IDX = {'I': range(0, 3),
                'a': range(3, 4),
                'all': range(0, 4),
                }

    parts = [parts] if type(parts) == str else parts

    result = []
    for part in parts:
        if part in DICT_IDX.keys():
            if dim == 3:
                squared = data[DICT_IDX[part], ...]**2
                axis = (0, 1)  # channel, N_freq
            else:
                squared = data[:, DICT_IDX[part], ...]**2
                axis = (1, 2)  # channel, N_freq

            if keep_freq_axis:
                axis = axis[:-1]
            norm = gen.sum_axis(squared, axis=axis)
            result.append(norm)
        else:
            raise ValueError('"parts" should be "I", "a", or "all" '
                             'or an array of them')

    return gen.stack(result)
