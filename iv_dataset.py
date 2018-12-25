import os
from copy import copy
from os import path
from typing import Dict, List, Sequence, Tuple, Union

import deepdish as dd
import numpy as np
import torch
from scipy.linalg import toeplitz
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import generic as gen
import mypath
from generic import TensArr, TensArrOrSeq
from normalize import LogMinMaxNormalization as NormalizationClass

SUFFIX = 'log_meanstd'
# ------determined by IV files------
NAME = dict(x='/IV_room', y='/IV_free',
            x_phase='/phase_room', y_phase='/phase_free',
            fname_wav='/fname_wav')
SEL_CH = {'all': dd.aslice[:, :, :],
          'alpha': dd.aslice[:, :, 3:4],
          True: None,
          }


class IVDataset(Dataset):
    """
    <Instance Variable>
    (not splitted)
    __PATH
    __needs
    __normconst

    (to be splitted)
    _all_files
    """

    __slots__ = ('__PATH', '__needs', '__normconst', '_all_files')

    def __init__(self, kind_data: str,
                 n_file=-1, random_sample=True, do_normalize=True, **kwargs):
        self.__PATH = mypath.path(f'iv_{kind_data}')
        self.__needs = dict(x='all', y='alpha', x_phase=False, y_phase=False, fname_wav=True)
        self.set_needs(**kwargs)

        # fname_list: The name of the file
        # that has information about data file list, mean, std, ...
        self._all_files: List[str]
        fname_list = path.join(self.__PATH, f'list_files_{n_file}_{SUFFIX}.h5')

        if path.isfile(fname_list):
            self._all_files, normconst = dd.io.load(fname_list)
            shouldsave = False

        else:
            # search all data files
            self._all_files = [f.path
                               for f in os.scandir(self.__PATH)
                               if f.name.endswith('.h5')
                               and not f.name.startswith('metadata')
                               and not f.name.startswith('list_files_')]
            self._all_files = sorted(self._all_files)
            if n_file != -1:
                if random_sample:
                    self._all_files = np.random.permutation(self._all_files)
                self._all_files = self._all_files[:n_file]
            shouldsave = True
            normconst = None

        if do_normalize:
            if not shouldsave:
                # if not normconst.get('consts'):
                #     normconst = dict(constnames=('mean', 'std'),
                #                      consts=(normconst['mean_x'],
                #                              normconst['mean_y'],
                #                              normconst['std_x'],
                #                              normconst['std_y'])
                #                      )
                #     shouldsave = True
                self.__normconst = NormalizationClass.load_from_dict(normconst)
            else:
                self.__normconst = NormalizationClass.create(self._all_files,
                                                             NAME['x'], NAME['y'])
        else:
            self.__normconst = None

        # if self._all_files[0].startswith('./Data'):
        #     self._all_files = [fname.replace('./Data', './data') for fname in self._all_files]
        #     shouldsave = True

        if shouldsave:
            dd.io.save(
                fname_list, (self._all_files,
                             self.__normconst.save_to_dict() if self.__normconst else None,
                             )
            )

        print(self.__normconst)
        print(f'{n_file} files prepared from {path.basename(self.__PATH)}.')

    def __len__(self):
        return len(self._all_files)

    def __getitem__(self, idx: int):
        sample = dict()
        for k, v in self.__needs.items():
            if v:
                data = dd.io.load(self._all_files[idx], group=NAME[k], sel=SEL_CH[v])
                if type(data) == np.ndarray:
                    data = gen.convert(data, astype=torch.Tensor)
                elif type(data) == np.str_:
                    data = str(data)
                sample[k] = data

        for xy in ('x', 'y'):
            sample[f'T_{xy}'] = sample[xy].shape[-2]

        return sample

    @staticmethod
    def pad_collate(batch: List[Dict]) -> Dict:
        result = dict()
        T_xs = np.array([item.pop('T_x') for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted]
        T_ys = np.array([batch[idx].pop('T_y') for idx in idxs_sorted])

        result['T_xs'], result['T_ys'] = T_xs, T_ys

        for key, value in batch[0].items():
            if type(value) != torch.Tensor:
                list_data = [batch[idx][key] for idx in idxs_sorted]
                set_data = set(list_data)
                if len(set_data) == 1:
                    result[key] = set_data.pop()
                else:
                    result[key] = list_data
            else:
                # B, T, F, C
                data = [batch[idx][key].permute(-2, -3, -1) for idx in idxs_sorted]
                data = pad_sequence(data, batch_first=True)
                # B, F, T, C
                data = data.permute(0, -2, -3, -1)

                result[key] = data

        return result

    def normalize_on_like(self, other):
        self.__normconst = other.__normconst

    def set_needs(self, **kwargs):
        """

        :param kwargs: dict(x: str, y: str, x_denorm: str, y_denorm: str, x_phase: str, y_phase: str)
        :return:
        """
        for k in self.__needs:
            if k in kwargs:
                assert kwargs[k] == 'all' or kwargs[k] == 'alpha' or type(kwargs[k]) == bool
                self.__needs[k] = kwargs[k]

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        return self.__normconst.normalize(a, xy)

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        return self.__normconst.normalize_(a, xy)

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        return self.__normconst.denormalize(a, xy)

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        return self.__normconst.denormalize_(a, xy)

    # noinspection PyProtectedMember
    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Tuple:
        """
        Split datasets.
        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: IVDataset
        :type ratio: Sequence[float]

        :rtype: Tuple[IVDataset]
        """
        if type(dataset) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(dataset._all_files),
                             dtype=int)
        result = [copy(dataset) for _ in range(n_split)]
        # all_f_per = np.random.permutation(a._all_files)

        for ii in range(n_split):
            result[ii]._all_files = dataset._all_files[idx_data[ii]:idx_data[ii + 1]]

        return tuple(result)


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


def delta(*data: TensArr, axis: int, L=2) -> TensArrOrSeq:
    dim = gen.ndim(data[0])
    if axis < 0:
        axis += dim

    max_len = max([item.shape[axis] for item in data])

    # Einsum expression
    # ex) if the member of a has the dim (b,c,f,t), (thus, axis=3)
    # einxp: ij,abcd -> abci
    str_axes = ''.join([chr(ord('a') + i) for i in range(dim)])
    str_new_axes = ''.join([chr(ord('a') + i) if i != axis else 'i'
                            for i in range(dim)])
    ein_expr = f'ij,{str_axes}->{str_new_axes}'

    # Create Toeplitz Matrix (T-2L, T)
    col = np.zeros(max_len - 2 * L, dtype=np.float32)
    col[0] = -L

    row = np.zeros(max_len, dtype=np.float32)
    row[:2 * L + 1] = range(-L, L + 1)

    denominator = np.sum([ll**2 for ll in range(1, L + 1)])
    tplz_mat = toeplitz(col, row) / (2 * denominator)

    # Convert to Tensor
    if type(data[0]) == torch.Tensor:
        if data[0].device == torch.device('cpu'):
            tplz_mat = torch.from_numpy(tplz_mat)
        else:
            tplz_mat = torch.tensor(tplz_mat, device=data[0].device)

    # Calculate
    result = [type(data[0])] * len(data)
    for idx, item in enumerate(data):
        length = item.shape[axis]
        result[idx] = gen.einsum(ein_expr, (tplz_mat[:length - 2 * L, :length], item))

    return result if len(result) > 1 else result[0]


DICT_IDX = {
    'I': range(0, 3),
    'a': range(-1, 0),
    'all': range(0, 4),
}


def norm_iv(data: TensArr,
            reduced_axis=(-3, -1),
            parts: Union[str, Sequence[str]] = 'all') -> TensArr:
    dim = gen.ndim(data)
    assert dim == 3 or dim == 4

    if data.shape[-1] == 1:
        parts = ('a',)
    else:
        parts = (parts,) if type(parts) == str else parts

    for part in parts:
        assert part in DICT_IDX

    str_axes = 'bftc' if dim == 4 else 'ftc'
    str_new_axes = str_axes
    for a in reduced_axis:
        str_new_axes = str_new_axes.replace(str_axes[a], '')

    ein_expr = f'{str_axes},{str_axes}->{str_new_axes}'

    result = [type(data)] * len(parts)
    for idx, part in enumerate(parts):
        result[idx] = gen.einsum(ein_expr, (data[..., DICT_IDX[part]],) * 2)

    return gen.stack(result)
