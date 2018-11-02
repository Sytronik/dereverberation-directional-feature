import pdb  # noqa: F401

import numpy as np
import deepdish as dd

import torch
from torch.utils.data import Dataset

from typing import Tuple, Any, Union, List, NamedTuple, Dict

import os
from os import path
from copy import copy
import multiprocessing as mp

import generic as gen
import mypath


class NormalizeConst(NamedTuple):
    mean_x: gen.TensArr
    mean_y: gen.TensArr
    std_x: gen.TensArr
    std_y: gen.TensArr

    def __str__(self):
        return (f'mean: {gen.shape(self.mean_x)}, {gen.shape(self.mean_y)},\t'
                f'std: {gen.shape(self.std_x)}, {gen.shape(self.std_y)}')

    @staticmethod
    def create_all(a, cuda_dev=1):
        if type(a[0]) == np.ndarray and a[0].dtype != np.float32:
            a = [item.astype(np.float32) for item in a]
        tensor_a \
            = [gen.convert(item, torch.Tensor) for item in a]
        return {
            'np': NormalizeConst(*a),
            'cpu': NormalizeConst(*tensor_a),
            'cuda': NormalizeConst(*[item.cuda(cuda_dev) for item in tensor_a])
        }


class IVDataset(Dataset):
    """
    <Instance Variable>
    (not splitted)
    PATH
    XNAME
    YNAME
    normalize

    (to be splitted)
    _all_files
    N_frames
    cum_N_frames
    """

    L_cut_x = 1
    L_cut_y = 1

    def __init__(self, split: str, XNAME: str, YNAME: str,
                 N_file=-1, doNormalize=True, cuda_dev=1):
        self.PATH = mypath.path(f'iv_{split}')
        self.XNAME = XNAME
        self.YNAME = YNAME

        # fname_list: The name of the file
        # that has information about data file list, mean, std, ...
        fname_list = path.join(self.PATH,
                               f'list_files_{N_file}_{IVDataset.L_cut_x}.h5')

        if path.isfile(fname_list):
            self._all_files, self.N_frames, self.cum_N_frames, normalize \
                = dd.io.load(fname_list)
            if doNormalize:
                self.normalize = NormalizeConst.create_all(normalize, cuda_dev)
                print(self.normalize['np'])
            else:
                self.normalize = None
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
                result = pool.map(IVDataset.sum_frames,
                                  [(fname, XNAME, YNAME)
                                   for fname in self._all_files])

                sum_x = np.sum([res[0] for res in result], axis=0)
                sum_y = np.sum([res[1] for res in result], axis=0)

                # sum_N_frame_x = np.sum([res[1] for res in result])
                self.N_frames = np.array([res[2] for res in result],
                                         dtype=np.int32)
                self.cum_N_frames = np.cumsum(self.N_frames)
                sum_N_frames = self.cum_N_frames[-1]

                # mean
                mean_x = sum_x / sum_N_frames
                mean_y = sum_y / sum_N_frames
                print(f'mean: {mean_x}, {mean_y},', end='\t')

                # Calculate Standard Deviation
                result = pool.map(IVDataset.sq_dev_frames,
                                  [(fname, XNAME, YNAME, mean_x, mean_y)
                                   for fname in self._all_files])

                pool.close()

                sum_sq_dev_x = np.sum([res[0] for res in result], axis=0)
                sum_sq_dev_y = np.sum([res[1] for res in result], axis=0)

                # stdev
                std_x = np.sqrt(sum_sq_dev_x / sum_N_frames + 1e-5)
                std_y = np.sqrt(sum_sq_dev_y / sum_N_frames + 1e-5)
                print(f'std: {std_x}, {std_y}')

                self.normalize \
                    = NormalizeConst.create_all((mean_x, mean_y, std_x, std_y),
                                                cuda_dev)
            else:
                # Calculate only the number of frames of each files
                self.N_frames = np.array(
                    pool.map(IVDataset.n_frame_files,
                             [(fname, YNAME) for fname in self._all_files]),
                    dtype=np.int32
                )
                pool.close()
                self.cum_N_frames = np.cumsum(self.N_frames)
                self.normalize = None

            dd.io.save(fname_list,
                       (self._all_files,
                        self.N_frames, self.cum_N_frames,
                        self.normalize['np'] if self.normalize else None)
                       )
        print(f'{len(self)} frames prepared '
              f'from {N_file} files of {path.basename(self.PATH)}.')

    @classmethod
    def n_frame_files(cls, tup: Tuple[str, str]) -> int:
        """
        (fname, YNAME) -> length of y
        """
        fname, YNAME = tup

        try:
            y = dd.io.load(fname, group='/' + YNAME)
        except:  # noqa: E722
            fname_npy = fname.replace('.h5', '.npy')
            data_dict = np.load(fname_npy).item()
            dd.io.save(fname, data_dict, compression=None)
            y = data_dict[YNAME]

        return y.shape[1] - cls.L_cut_x // 2

    @classmethod
    def sum_frames(cls, tup: Tuple[str, str, str]) -> Tuple[Any, Any, int]:
        """
        (fname, XNAME, YNAME) -> (sum of x, length of x, sum of y, length of y)
        """
        fname, XNAME, YNAME = tup

        try:
            data_dict = dd.io.load(fname)
        except:  # noqa: E722
            raise Exception(fname)

        half = cls.L_cut_x // 2
        x = data_dict[XNAME]
        y = data_dict[YNAME]
        max_idx = y.shape[1]
        min_idx = half
        x_stacked = cls.stack_x(x[:, min_idx - half:max_idx + half + 1, :])
        y = y[:, min_idx:max_idx, :]

        return (x_stacked.sum(axis=0),
                y.sum(axis=1)[:, np.newaxis, :],
                y.shape[1])

    @classmethod
    def sq_dev_frames(cls,
                      tup: Tuple[str, str, str, Any, Any]) -> Tuple[Any, Any]:
        """
        (fname, XNAME, YNAME, mean of x, mean of y)
        -> (sum of deviation of x, sum of deviation of y)
        """
        fname, XNAME, YNAME, mean_x, mean_y = tup
        data_dict = dd.io.load(fname)
        half = cls.L_cut_x // 2
        x = data_dict[XNAME]
        y = data_dict[YNAME]
        max_idx = y.shape[1]
        min_idx = half
        x_stacked = cls.stack_x(x[:, min_idx - half:max_idx + half + 1, :])
        y = y[:, min_idx:max_idx, :]

        return (((x_stacked - mean_x)**2).sum(axis=0),
                ((y - mean_y)**2).sum(axis=1)[:, np.newaxis, :],
                )

    def doNormalize(self, const: Dict[str, NormalizeConst], cuda_dev=1):
        self.normalize = const

    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                normalize = self.normalize['cpu']
            else:
                normalize = self.normalize['cuda']
        else:
            normalize = self.normalize['np']

        if xy == 'x':
            return normalize.std_x * a + normalize.mean_x
        elif xy == 'y':
            return normalize.std_y * a + normalize.mean_y
        else:
            raise 'xy should be "x" or "y"'

    def denormalize_(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                normalize = self.normalize['cpu']
            else:
                normalize = self.normalize['cuda']
        else:
            normalize = self.normalize['np']

        if xy == 'x':
            return a.mul_(normalize.std_x).add_(normalize.mean_x)
        elif xy == 'y':
            return a.mul_(normalize.std_y).add_(normalize.mean_y)
        else:
            raise 'xy should be "x" or "y"'

    def __len__(self):
        return self.cum_N_frames[-1]

    def __getitem__(self, idx: int):
        half = IVDataset.L_cut_x // 2
        # File Index
        i_file = np.where(self.cum_N_frames > idx)[0][0]

        # Frame Index of y
        if i_file >= 1:
            i_frame = idx - self.cum_N_frames[i_file - 1] + half
        else:
            i_frame = idx + half
        if i_frame >= self.N_frames[i_file] + half:
            pdb.set_trace()

        range_x = range(i_frame - half, i_frame + half + 1)

        # File Open (with Slicing)
        x = dd.io.load(self._all_files[i_file],
                       group='/' + self.XNAME,
                       sel=dd.aslice[:, range_x, :]
                       )
        y = dd.io.load(self._all_files[i_file],
                       group='/' + self.YNAME,
                       sel=dd.aslice[:, i_frame:i_frame + 1, :]
                       )

        # Normalize
        if self.normalize:
            x = (x - self.normalize['np'].mean_x) / self.normalize['np'].std_x
            y = (y - self.normalize['np'].mean_y) / self.normalize['np'].std_y

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        sample = {'x': x, 'y': y}

        return sample

    @classmethod
    def stack_x(cls, x: gen.TensArr) -> gen.TensArr:
        """
        Make groups of the frames of x and stack the groups
        x_stacked: (time_length) x (N_freq) x (L_cut_x) x (XYZ0 channel)
        """
        if cls.L_cut_x == 1:
            return x
        if gen.ndim(x) != 3:
            raise Exception('Dimension Mismatch')

        return gen.stack([x[:, ii:ii + cls.L_cut_x, :]
                          for ii in range(x.shape[1] - cls.L_cut_x)
                          ])
    #
    # @classmethod
    # def stack_y(cls, y: gen.TensArr) -> gen.TensArr:
    #     """
    #     y_stacked: (time_length) x (N_freq) x (1) x (XYZ0 channel)
    #     """
    #     if gen.ndim(y) != 3:
    #         raise Exception('Dimension Mismatch')
    #
    #     return gen.transpose(y, (1, 0, 2))[:, :, None, :]

    @classmethod
    def unstack_x(cls, x: gen.TensArr) -> gen.TensArr:
        if gen.ndim(x) != 4 or gen.shape(x)[2] <= cls.L_cut_x // 2:
            raise Exception('Dimension/Size Mismatch')

        return gen.transpose(x[:, :, cls.L_cut_x // 2, :], (1, 0, 2))

    @classmethod
    def unstack_y(cls, y: gen.TensArr) -> gen.TensArr:
        if gen.ndim(y) != 4 or gen.shape(y)[2] != 1:
            raise Exception('Dimension/Size Mismatch')

        return gen.transpose(gen.squeeze(y, axis=2), (1, 0, 2))

    # @staticmethod
    # def my_collate(batch):
    #     x_stacked = torch.cat([item['x_stacked'] for item in batch])
    #     y_stacked = torch.cat([item['y_stacked'] for item in batch])
    #     return (x_stacked, y_stacked)

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
            result[ii].N_frames \
                = a.N_frames[idx_data[ii]:idx_data[ii + 1]]
            result[ii].cum_N_frames \
                = np.cumsum(result[ii].N_frames)

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

    if type(parts) == str:
        parts = [parts]

    result = []
    for part in parts:
        if part in DICT_IDX.keys():
            squared = data[..., DICT_IDX[part]]**2
            if dim == 3:
                axis = (0, 2)  # N_freq, channel
            else:
                axis = (1, 2, 3)  # N_freq, L_cut, channel
            if keep_freq_axis:
                axis = axis[1:]
            norm = gen.sum_axis(squared, axis=axis)
            if dim == 3:
                norm = gen.transpose(norm)
            result.append(norm)
        else:
            raise ValueError('"parts" should be "I", "a", or "all" '
                             'or an array of them')

    return result[0] if len(result) == 1 else gen.stack(result)
