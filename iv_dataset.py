from copy import copy
import os
from os import path
from typing import Tuple, Union, List, Dict, Iterable

import deepdish as dd
import numpy as np
from scipy.linalg import toeplitz

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import generic as gen
from generic import TensArr
import mypath
from normalize import LogMeanStdNormalization as NormalizationClass


SUFFIX = 'log'


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

    def __init__(self, kind_data: str, xname: str, yname: str,
                 N_file=-1, doNormalize=True):
        self.PATH = mypath.path(f'iv_{kind_data}')
        self.XNAME = xname
        self.YNAME = yname

        # fname_list: The name of the file
        # that has information about data file list, mean, std, ...
        fname_list = path.join(self.PATH, f'list_files_{N_file}_{SUFFIX}.h5')

        if path.isfile(fname_list):
            self._all_files, normconst = dd.io.load(fname_list)
            shouldsave = False
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
            shouldsave = True

        if doNormalize:
            if normconst:
                self.__normconst = NormalizationClass.load_from_dict(normconst)
            else:
                self.__normconst = NormalizationClass.create(self._all_files, xname, yname)
        else:
            self.__normconst = None

        if shouldsave:
            dd.io.save(
                fname_list, (self._all_files,
                             self.__normconst.save_to_dict() if self.__normconst else None,
                             )
            )

        print(self.__normconst)
        print(f'{N_file} files prepared from {path.basename(self.PATH)}.')

    def __len__(self):
        return len(self._all_files)

    def __getitem__(self, idx: int):
        # File Open (with Slicing)
        data_dict = dd.io.load(self._all_files[idx])
        x = data_dict[self.XNAME]
        y = data_dict[self.YNAME]

        # Normalize
        if self.__normconst:
            x = self.__normconst.normalize(x, 'x')
            y = self.__normconst.normalize(y, 'y')

        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        # y = F.pad(y, (0, x.size()[-1]-y.sisz()[-1]))

        sample = {'x': x, 'y': y,
                  'T_x': x.shape[1], 'T_y': y.shape[1],
                  # 'fname': self._all_files[idx],
                  }

        return sample

    def pad_collate(self, batch: List) -> Dict:
        T_xs = np.array([item['T_x'] for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted]
        T_ys = np.array([batch[idx]['T_y'] for idx in idxs_sorted])

        x = [batch[idx]['x'].permute(-2, -3, -1) for idx in idxs_sorted]
        y = [batch[idx]['y'].permute(-2, -3, -1) for idx in idxs_sorted]

        x = pad_sequence(x, batch_first=True)
        y = pad_sequence(y, batch_first=True)

        x = x.permute(0, -2, -3, -1)
        y = y.permute(0, -2, -3, -1)

        # fnames = [i1tem['fname'] for item in batch]
        return {'x': x, 'y': y,
                'T_xs': T_xs, 'T_ys': T_ys,
                # 'fnames': fnames,
                }

    def normalize_on_like(self, other):
        self.__normconst = other.__normconst

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        return self.__normconst.denormalize(a, xy)

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        return self.__normconst.denormalize_(a, xy)

    @classmethod
    def split(cls,
              a, ratio: Iterable[float]) -> Tuple:
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
            raise Exception("Only one element of the parameter 'ratio' can have the value of -1")
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


def delta(a, axis: int, L=2, tplz_mat=None):
    if axis < 0:
        axis += gen.ndim(a)
    str_axes = ''.join([chr(ord('a') + i)
                        for i in range(gen.ndim(a))])
    str_new_axes = ''.join([chr(ord('a') + i)
                            if i != axis else 'i' for i in range(gen.ndim(a))])
    einexp = f'ij,{str_axes}->{str_new_axes}'

    if tplz_mat is None:
        tplz_mat = np.zeros((a.shape[axis] - 2 * L, a.shape[axis]))
        col = np.zeros(a.shape[axis] - 2 * L)
        row = np.zeros(a.shape[axis])
        col[0] = -1
        row[:L] = -1
        row[L + 1:2 * L + 1] = 1

        denominator = np.sum([ll**2 for ll in range(1, L + 1)])
        tplz_mat = toeplitz(col, row) / (2 * denominator)

        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                tplz_mat = gen.convert(tplz_mat, torch.Tensor)
            else:
                tplz_mat = gen.convert(tplz_mat, torch.Tensor).cuda(device=a.device)

    a = gen.einsum(einexp, (tplz_mat, a))

    return a, tplz_mat


DICT_IDX = {
    'I': range(0, 3),
    'a': range(3, 4),
    'all': range(0, 4),
}


def norm_iv(data: TensArr, reduced_axis=(-3, -1),
            parts: Union[str, Iterable[str]]='all') -> TensArr:
    dim = gen.ndim(data)
    if dim != 3 and dim != 4:
        raise f'Dimension Mismatch: {dim}'

    parts = [parts] if type(parts) == str else parts
    base_expr = 'bftc' if dim == 4 else 'ftc'
    result_expr = ''.join([base_expr[a] for a in reduced_axis])

    ein_expr = f'{base_expr},{base_expr}->{result_expr}'

    result = []
    for part in parts:
        if part in DICT_IDX.keys():
            norm = gen.einsum(ein_expr, (data[..., DICT_IDX[part]],) * 2)
            result.append(norm)
        else:
            raise ValueError('"parts" should be "I", "a", or "all" '
                             'or an array of them')

    return gen.stack(result)


# def delta_no_tplz(a, axis: int, L=2):
#     dims = list(range(a.dim()))
#     dims[0], dims[axis] = dims[axis], dims[0]
#     a = a.permute(dims)
#
#     diffs = [l*(a[2*l:] - a[:-2*l])[2*(L-l):len(a)-2*(L-l)]
#              for l in range(1, L+1)]
#     a = sum(diffs) / (sum([l**2 for l in range(1, L+1)]) * 2)
#     a = a.permute(dims)
#     return a
