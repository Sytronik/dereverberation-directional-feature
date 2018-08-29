import pdb  # noqa: F401

import numpy as np
import deepdish as dd

import torch
from torch.utils.data import Dataset

from typing import Tuple, Any, Union, List, NamedTuple

import os
from os import path
from copy import copy
import multiprocessing as mp

import generic as gen


class NormalizeConst(NamedTuple):
    mean_x: gen.TensArr
    mean_y: gen.TensArr
    std_x: gen.TensArr
    std_y: gen.TensArr

    def __str__(self):
        return (f'mean: {self.mean_x.shape}, {self.mean_y.shape},\t'
                f'std: {self.std_x.shape}, {self.std_y.shape}')

    def astype(self, T: type=torch.Tensor):
        return NormalizeConst(*[gen.convert(item, T) for item in self])


class IVDataset(Dataset):
    """
    <Instance Variable>
    (not splitted)
    DIR
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

    def __init__(self, DIR: str, XNAME: str, YNAME: str,
                 N_file=-1, doNormalize=True):
        self.DIR = DIR
        self.XNAME = XNAME
        self.YNAME = YNAME

        # fname_list: The name of the file
        # that has information about data file list, mean, std, ...
        fname_list \
            = path.join(DIR, f'list_files_{N_file}_{IVDataset.L_cut_x}.h5')
        if path.isfile(fname_list):
            self._all_files, self.N_frames, self.cum_N_frames, normalize \
                = dd.io.load(fname_list)

            if doNormalize:
                self.normalize = NormalizeConst(*normalize)
            else:
                self.normalize = None
            print(self.normalize)
        else:
            # search all data files
            _all_files = [f.path
                          for f in os.scandir(DIR)
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

                self.normalize = NormalizeConst(mean_x, mean_y, std_x, std_y)
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
                        self.normalize)
                       )
        print(f'{len(self)} frames prepared '
              f'from {N_file} files of {path.basename(DIR)}.')

    @classmethod
    def n_frame_files(cls, tup: Tuple[str, str]) -> int:
        """
        (fname, YNAME) -> length of y
        """
        fname, YNAME = tup

        try:
            y = dd.io.load(fname, group='/'+YNAME)
        except:  # noqa: E722
            fname_npy = fname.replace('.h5', '.npy')
            data_dict = np.load(fname_npy).item()
            dd.io.save(fname, data_dict, compression=None)
            y = data_dict[YNAME]

        # x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return y.shape[1]

    @classmethod
    def sum_frames(cls, tup: Tuple[str, str, str]) -> Tuple[Any, Any, int]:
        """
        (fname, XNAME, YNAME) -> (sum of x, length of x, sum of y, length of y)
        """
        fname, XNAME, YNAME = tup

        try:
            data_dict = dd.io.load(fname)
        except:  # noqa: E722
            fname_npy = fname.replace('.h5', '.npy')
            data_dict = np.load(fname_npy).item()
            dd.io.save(fname, data_dict, compression=None)

        x = data_dict[XNAME]
        y = data_dict[YNAME]

        x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return (x_stacked.sum(axis=0), y.sum(axis=1)[:, np.newaxis, :],
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
        x = data_dict[XNAME]
        y = data_dict[YNAME]

        x_stacked = cls.stack_x(x, L_trunc=y.shape[1])

        return (((x_stacked - mean_x)**2).sum(axis=0),
                ((y - mean_y)**2).sum(axis=1)[:, np.newaxis, :],
                )

    def doNormalize(self, const: NormalizeConst):
        self.normalize = const

    def denormalize(self, a: gen.TensArr, xy: str) -> gen.TensArr:
        normalize = self.normalize.astype(type(a))
        if xy == 'x':
            return normalize.std_x * a + normalize.mean_x
        elif xy == 'y':
            return normalize.std_y * a + normalize.mean_y
        else:
            raise 'xy should be "x" or "y"'

    def __len__(self):
        return self.cum_N_frames[-1]

    def __getitem__(self, idx: int):
        # File Index
        i_file = np.where(self.cum_N_frames > idx)[0][0]

        # Frame Index of y
        if i_file >= 1:
            i_frame = idx - self.cum_N_frames[i_file-1]
        else:
            i_frame = idx
        if i_frame >= self.N_frames[i_file]:
            pdb.set_trace()

        # Frame range of stacked x
        i_x_lower = i_frame - IVDataset.L_cut_x//2
        if i_x_lower < 0:
            margin_lower = 0 - i_x_lower
        else:
            margin_lower = 0
        i_x_lower = i_x_lower+margin_lower

        i_x_upper = i_frame + IVDataset.L_cut_x//2 + 1
        if i_x_upper > self.N_frames[i_file]:
            margin_upper = i_x_upper - self.N_frames[i_file]
        else:
            margin_upper = 0
        i_x_upper = i_x_upper-margin_upper

        # File Open (with Slicing)
        x_stacked = dd.io.load(self._all_files[i_file],
                               group='/'+self.XNAME,
                               sel=dd.aslice[:, i_x_lower:i_x_upper, :]
                               )
        y_stacked = dd.io.load(self._all_files[i_file],
                               group='/'+self.YNAME,
                               sel=dd.aslice[:, i_frame:i_frame+1, :]
                               )

        # Zero-padding & unsqueeze
        L0, _, L2 = x_stacked.shape
        x_stacked = np.concatenate((np.zeros((L0, margin_lower, L2)),
                                    x_stacked,
                                    np.zeros((L0, margin_upper, L2))),
                                   axis=1)
        # data_dict = np.load(self._all_files[idx]).item()
        # x = data_dict[self.XNAME]
        # y = data_dict[self.YNAME]

        # Stack & Normalize
        # x_stacked = IVDataset.stack_x(x, L_trunc=y.shape[1])
        if self.normalize:
            x_stacked = (x_stacked-self.normalize.mean_x)/self.normalize.std_x
            y_stacked = (y_stacked-self.normalize.mean_y)/self.normalize.std_y

        x_stacked = torch.from_numpy(x_stacked).float()
        y_stacked = torch.from_numpy(y_stacked).float()
        sample = {'x_stacked': x_stacked, 'y_stacked': y_stacked}

        return sample

    @classmethod
    def stack_x(cls, x: gen.TensArr, L_trunc=0) -> gen.TensArr:
        """
        Make groups of the frames of x and stack the groups
        x_stacked: (time_length) x (N_freq) x (L_cut_x) x (XYZ0 channel)
        """
        if cls.L_cut_x == 1:
            return x
        if gen.ndim(x) != 3:
            raise Exception('Dimension Mismatch')

        half = cls.L_cut_x//2

        L0, L1, L2 = gen.shape(x)

        x = gen.cat((np.zeros((L0, half, L2)),
                     x,
                     np.zeros((L0, half, L2))),
                    axis=1, astype=type(x))

        if L_trunc != 0:
            L1 = L_trunc

        return gen.stack([x[:, ii - half:ii + half + 1, :]
                          for ii in range(half, half + L1)
                          ])

    @classmethod
    def stack_y(cls, y: gen.TensArr) -> gen.TensArr:
        """
        y_stacked: (time_length) x (N_freq) x (1) x (XYZ0 channel)
        """
        if gen.ndim(y) != 3:
            raise Exception('Dimension Mismatch')

        return gen.transpose(y, (1, 0, 2))[:, :, None, :]

    @classmethod
    def unstack_x(cls, x: gen.TensArr) -> gen.TensArr:
        if gen.ndim(x) != 4 or gen.shape(x)[2] <= cls.L_cut_x//2:
            raise Exception('Dimension/Size Mismatch')

        x = x[:, :, cls.L_cut_x//2, :]

        return gen.transpose(x, (1, 0, 2))

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
                = a._all_files[idx_data[ii]:idx_data[ii+1]]
            result[ii].N_frames \
                = a.N_frames[idx_data[ii]:idx_data[ii+1]]
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
                norm = transpose(norm)
            result.append(norm)
        else:
            raise ValueError('"parts" should be "I", "a", or "all" '
                             'or an array of them')

    return result[0] if len(result) == 1 else stack(result)
