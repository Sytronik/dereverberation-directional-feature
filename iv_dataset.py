import pdb  # noqa: F401

import numpy as np
import deepdish as dd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from typing import Tuple, Union, List, NamedTuple, Dict

import os
from os import path
from copy import copy
import multiprocessing as mp

import generic as gen
import mypath
from utils import static_vars


class NormalizeConst(NamedTuple):
    mean_x: Union[gen.TensArr, np.float64]
    mean_y: Union[gen.TensArr, np.float64]
    std_x: Union[gen.TensArr, np.float64]
    std_y: Union[gen.TensArr, np.float64]
    min_x: Union[gen.TensArr, np.float64]
    min_y: Union[gen.TensArr, np.float64]

    def normalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std, min_ = self._get_items_as_type_of(a, xy)
        # return (a - mean) / std - min_
        return (a - mean) / std

    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std, min_ = self._get_items_as_type_of(a, xy)
        # return (a + min_) * std + mean
        return a * std + mean

    def denormalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        mean, std, min_ = self._get_items_as_type_of(a, xy)
        # a += min_
        a *= std
        a += mean
        return a

    def convert(self, astype, device=torch.device('cpu')):
        if astype == torch.Tensor:
            members = [gen.convert(item, astype).to(device) for item in self]
        else:
            members = [gen.convert(item, astype) for item in self]
        return NormalizeConst(*members)

    def _get_items_only(self, xy: str):
        try:
            mean = eval(f'self.mean_{xy}')
            std = eval(f'self.std_{xy}')
            min_ = eval(f'self.min_{xy}')
        except AttributeError:
            raise Exception('xy should be "x" or "y"')

        return mean, std, min_

    @static_vars(_cpu=None, _cuda=None)
    def _get_items_as_type_of(self, a: gen.TensArr, xy: str):
        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                if not self._get_items_as_type_of._cpu:
                    self._get_items_as_type_of._cpu \
                        = self.convert(type(a), device=a.device)
                    mean, std, min_ \
                        = self._get_items_as_type_of._cpu._get_items_only(xy)
            else:
                if not self._get_items_as_type_of._cuda:
                    self._get_items_as_type_of._cuda \
                        = self.convert(type(a), device=a.device)
                mean, std, min_ \
                    = self._get_items_as_type_of._cuda._get_items_only(xy)
        else:
            mean, std, min_ = self._get_items_only(xy)

        if std.shape[0] == a.shape[-2]:
            std = std.permute((2, 0, 1))
        return mean, std, min_

    def __str__(self):
        return (f'mean: {self.mean_x}, {self.mean_y}, '
                f'std: {self.std_x}, {self.std_y}, '
                f'min: {self.min_x}, {self.min_y}')


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
        fname_list = path.join(self.PATH,
                               f'list_files_{N_file}_no_cut.h5')

        if path.isfile(fname_list):
            self._all_files, normconst = dd.io.load(fname_list)
            if doNormalize:
                self._normconst_np = NormalizeConst(*normconst)
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

            pool = mp.Pool(mp.cpu_count())
            if doNormalize:
                # Calculate summation & no. of total frames (parallel)
                result = pool.map(IVDataset.calc_min_sum_size,
                                  [(fname, XNAME, YNAME)
                                   for fname in self._all_files])

                min_x = np.min([res[0]['min'] for res in result])
                min_y = np.min([res[1]['min'] for res in result])
                # sum_x = np.sum([item[0]['sum'] for item in result])
                # sum_y = np.sum([item[1]['sum'] for item in result])
                sum_size_x = np.sum([item[0]['size'] for item in result])
                sum_size_y = np.sum([item[1]['size'] for item in result])

                mean_x = 0.  # sum_x / sum_size_x
                mean_y = 0.  # sum_y / sum_size_y

                result = pool.map(IVDataset.calc_sq_dev,
                                  [(fname, XNAME, YNAME, mean_x, mean_y)
                                   for fname in self._all_files])
                pool.close()

                sum_sq_dev_x = np.sum([item[0] for item in result])
                sum_sq_dev_y = np.sum([item[1] for item in result])
                sum_size_x //= sum_sq_dev_x.size
                sum_size_y //= sum_sq_dev_y.size
                std_x = np.sqrt(sum_sq_dev_x / sum_size_x + 1e-5)
                std_y = np.sqrt(sum_sq_dev_y / sum_size_y + 1e-5)
                # min_x = np.min([res[0][0] for res in result])
                # min_y = np.min([res[1][0] for res in result])
                #
                # max_x = np.max([res[0][1] for res in result])
                # max_y = np.max([res[1][1] for res in result])
                # self._normconst_np = NormalizeConst(min_x, min_y,
                #                                     max_x, max_y)
                self._normconst_np = NormalizeConst(mean_x, mean_y,
                                                    std_x, std_y,
                                                    min_x, min_y)
            else:
                self._normconst_np = None

            dd.io.save(fname_list,
                       (self._all_files,
                        self._normconst_np if self._normconst_np else None,
                        )
                       )
        print(f'{N_file} files prepared from {path.basename(self.PATH)}.')

    @staticmethod
    def calc_min_sum_size(tup: Tuple[str, str, str]) -> Tuple[Dict, Dict]:
        """
        (fname, XNAME, YNAME) -> (sum of x, length of x, sum of y, length of y)
        """
        fname, XNAME, YNAME = tup

        try:
            data_dict = dd.io.load(fname)
        except:  # noqa: E722
            raise Exception(fname)
        x, y = data_dict[XNAME], data_dict[YNAME]

        ss_x = {'min': x.min(),
                # 'sum': x.sum(),
                'size': x.size}
        ss_y = {'min': y.min(),
                # 'sum': y.sum(),
                'size': y.size}

        print('.', end='', flush=True)
        return (ss_x, ss_y)

    @staticmethod
    def calc_sq_dev(tup: Tuple[str, str, str, float, float]) \
            -> Tuple[Dict, Dict]:
        """
        (fname, XNAME, YNAME, mean of x, mean of y)
        -> (sum of deviation of x, sum of deviation of y)
        """
        fname, XNAME, YNAME, mean_x, mean_y = tup

        data_dict = dd.io.load(fname)
        x, y = data_dict[XNAME], data_dict[YNAME]

        sq_dev_x = ((x - mean_x)**2).sum(axis=(1, 2), keepdim=True)
        sq_dev_y = ((y - mean_y)**2).sum(axis=(1, 2), keepdim=True)

        print('.', end='', flush=True)
        return (sq_dev_x, sq_dev_y)

    # @staticmethod
    # def minmax(tup: Tuple[str, str, str]) -> Tuple[Tuple, Tuple]:
    #     """
    #     (fname, XNAME, YNAME)->(sum of x, length of x, sum of y, length of y)
    #     """
    #     fname, XNAME, YNAME = tup
    #
    #     try:
    #         data_dict = dd.io.load(fname)
    #     except:  # noqa: E722
    #         raise Exception(fname)
    #
    #     x, y = data_dict[XNAME], data_dict[YNAME]
    #     mm_x = (x.min(), x.max())
    #     mm_y = (y.min(), y.max())
    #     print('.', end='', flush=True)
    #     return (mm_x, mm_y)

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

        sample = {'x': x, 'y': y, 'fname': self._all_files[idx]}

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

        pad_value_x \
            = self._normconst_np.normalize(0, 'x') if self._normconst_np else 0
        pad_value_y \
            = self._normconst_np.normalize(0, 'y') if self._normconst_np else 0

        for idx, item in enumerate(batch):
            x[idx] = F.pad(item['x'],
                           (0, int(max_N_frame - N_frames_x[idx])),
                           value=pad_value_x)
            y[idx] = F.pad(item['y'],
                           (0, int(max_N_frame - N_frames_y[idx])),
                           value=pad_value_y)

        fnames = [item['fname'] for item in batch]
        return {'x': torch.stack(x), 'y': torch.stack(y),
                'N_frames_x': N_frames_x, 'N_frames_y': N_frames_y,
                'fnames': fnames,
                }

    def normalizeOn(self, const: Dict[str, NormalizeConst]):
        self._normconst_np = const

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
