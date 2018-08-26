import pdb  # noqa: F401

import numpy as np
import deepdish as dd

import torch
from torch.utils.data import Dataset

from typing import Tuple, Any, Union, List, NamedTuple

import os
from copy import copy
import multiprocessing as mp


TensorArray = Union[torch.Tensor, np.ndarray]


def convert(a: TensorArray, astype: type) -> TensorArray:
    if astype == torch.Tensor:
        if type(a) == torch.Tensor:
            return a
        else:
            return torch.tensor(a, dtype=torch.float32)
    elif astype == np.ndarray:
        if type(a) == torch.Tensor:
            return a.numpy()
        else:
            return a
    else:
        raise ValueError(astype)


def shape(a: TensorArray) -> Tuple:
    if type(a) == torch.Tensor:
        return tuple(a.size())
    elif type(a) == np.ndarray:
        return a.shape
    else:
        raise TypeError


def ndim(a: TensorArray) -> int:
    if type(a) == torch.Tensor:
        return a.dim()
    elif type(a) == np.ndarray:
        return a.ndim
    else:
        raise TypeError


def transpose(a: TensorArray, axes: Union[Tuple, int]=None) -> TensorArray:
    if type(a) == torch.Tensor:
        if not axes:
            if a.dim() >= 2:
                return a.permute((1, 0)+(-1,)*(a.dim()-2))
            else:
                return a
        else:
            return a.permute(axes)

    elif type(a) == np.ndarray:
        if a.ndim == 1 and not axes:
            return a
        else:
            return a.transpose(axes)
    else:
        raise TypeError


def squeeze(a: TensorArray, axis=None) -> int:
    if type(a) == torch.Tensor:
        return a.squeeze(dim=axis)
    elif type(a) == np.ndarray:
        return a.squeeze(axis=axis)
    else:
        raise TypeError


def _cat_stack(fn: str,
               a: Union[List, Tuple],
               axis=0,
               astype: type=None) -> TensorArray:
    fn_dict = {(torch, 'cat'): torch.cat,
               (np, 'cat'): np.concatenate,
               (torch, 'stack'): torch.stack,
               (np, 'stack'): np.stack,
               }

    types = [type(item) for item in a]
    if np.any(types != types[0]):
        a = [convert(item, (astype if astype else types[0])) for item in a]

    if types[0] == torch.Tensor:
        result = fn_dict[(torch, fn)](a, dim=axis)
    elif types[0] == np.ndarray:
        result = fn_dict[(np, fn)](a, axis=axis)
    else:
        raise TypeError

    return convert(result, astype) if astype else result


def cat(*args, **kargs) -> TensorArray:
    return _cat_stack('cat', *args, **kargs)


def stack(*args, **kargs) -> TensorArray:
    return _cat_stack('stack', *args, **kargs)


def sum_axis(a: TensorArray, axis=None):
    if axis:
        if type(a) == torch.Tensor:
            return a.sum(dim=axis)
        elif type(a) == np.ndarray:
            return a.sum(axis=axis)
        else:
            raise TypeError
    else:
        return a.sum()


class NormalizeConst(NamedTuple):
    mean_x: TensorArray
    mean_y: TensorArray
    std_x: TensorArray
    std_y: TensorArray

    def __str__(self):
        return (f'mean: {self.mean_x.shape}, {self.mean_y.shape},\t'
                f'std: {self.std_x.shape}, {self.std_y.shape}')

    def astype(self, T: type=torch.Tensor):
        return NormalizeConst(*[convert(item, T) for item in self])


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
                 N_data=-1, doNormalize=True):
        self.DIR = DIR
        self.XNAME = XNAME
        self.YNAME = YNAME

        fname_list = os.path.join(DIR, f'list_files_{N_data}.h5')
        if os.path.isfile(fname_list):
            self._all_files, self.N_frames, self.cum_N_frames, normalize \
                = dd.io.load(fname_list)

            if doNormalize:
                self.normalize = NormalizeConst(*normalize)
            else:
                self.normalize = None
            print(self.normalize)
        else:
            _all_files = [f.path
                          for f in os.scandir(DIR)
                          if f.name.endswith('.h5')
                          and f.name != 'metadata.h5'
                          and not f.name.startswith('list_files_')]
            self._all_files = np.random.permutation(_all_files)
            if N_data != -1:
                self._all_files = self._all_files[:N_data]

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
              f'from {N_data} files of {os.path.basename(DIR)}.')

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

    def denormalize(self, a: TensorArray, xy: str) -> TensorArray:
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
    def stack_x(cls, x: TensorArray, L_trunc=0) -> TensorArray:
        """
        Make groups of the frames of x and stack the groups
        x_stacked: (time_length) x (N_freq) x (L_cut_x) x (XYZ0 channel)
        """
        if cls.L_cut_x == 1:
            return x
        if ndim(x) != 3:
            raise Exception('Dimension Mismatch')

        half = cls.L_cut_x//2

        L0, L1, L2 = shape(x)

        x = cat((np.zeros((L0, half, L2)),
                 x,
                 np.zeros((L0, half, L2))),
                axis=1, astype=type(x))

        if L_trunc != 0:
            L1 = L_trunc

        return stack([x[:, ii - half:ii + half + 1, :]
                      for ii in range(half, half + L1)
                      ])

    @classmethod
    def stack_y(cls, y: TensorArray) -> TensorArray:
        """
        y_stacked: (time_length) x (N_freq) x (1) x (XYZ0 channel)
        """
        if ndim(y) != 3:
            raise Exception('Dimension Mismatch')

        return transpose(y, (1, 0, 2))[:, :, None, :]

    @classmethod
    def unstack_x(cls, x: TensorArray) -> TensorArray:
        if ndim(x) != 4 or shape(x)[2] <= cls.L_cut_x//2:
            raise Exception('Dimension/Size Mismatch')

        x = x[:, :, cls.L_cut_x//2, :]

        return transpose(x, (1, 0, 2))

    @classmethod
    def unstack_y(cls, y: TensorArray) -> TensorArray:
        if ndim(y) != 4 or shape(y)[2] != 1:
            raise Exception('Dimension/Size Mismatch')

        return transpose(squeeze(y, axis=2), (1, 0, 2))

    # @staticmethod
    # def my_collate(batch):
    #     x_stacked = torch.cat([item['x_stacked'] for item in batch])
    #     y_stacked = torch.cat([item['y_stacked'] for item in batch])
    #     return (x_stacked, y_stacked)

    @classmethod
    def split(cls, a, ratio: Tuple) -> Tuple:
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

        return result


def norm_iv(data: TensorArray, parts: Union[str, List[str], Tuple[str]]='all'):
    dim = ndim(data)
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
            temp = data[..., DICT_IDX[part]]**2
            if dim == 3:
                axis = (0, 2)
            else:
                axis = (1, 2, 3)
            result.append(sum_axis(temp, axis=axis))
        else:
            raise ValueError('"parts" should be "I", "a", or "all" '
                             'or an array of them')

    return result[0] if len(result) == 1 else result
