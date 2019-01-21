import os
from copy import copy
from typing import Dict, List, Sequence, Tuple

import deepdish as dd
import numpy as np
import torch
import scipy.io as scio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from generic import TensArr
import config as cfg
# noinspection PyUnresolvedReferences
import normalize


class IVDataset(Dataset):
    """
    <Instance Variables>
    (not splitted)
    _PATH
    _needs
    _normconst

    (to be splitted)
    _all_files
    """

    __slots__ = ('_PATH', '_needs', '_normconst', '_all_files')

    def __init__(self, kind_data: str,
                 n_file=-1, random_by_utterance=False, norm_class=None, **kwargs):
        self._PATH = cfg.DICT_PATH[f'iv_{kind_data}']
        self._needs = dict(x='all', y='alpha',
                           x_phase=False, y_phase=False,
                           fname_wav=True)
        self.set_needs(**kwargs)

        # f_normconst: The name of the file
        # that has information about data file list, mean, std, ...
        f_normconst = cfg.DICT_PATH[f'normconst_{kind_data}'].format(n_file)

        if os.path.isfile(f_normconst):
            _all_files, normconst = dd.io.load(f_normconst)
            shouldsave = False
        else:
            # search all data files
            _all_files = [f.name for f in os.scandir(self._PATH) if cfg.is_ivfile(f)]
            _all_files = sorted(_all_files)
            if n_file != -1:
                if random_by_utterance:
                    utterances = np.random.randint(len(_all_files) // cfg.N_LOC[kind_data],
                                                   size=n_file // cfg.N_LOC[kind_data])
                    utterances = [f'{u:4d}_' for u in utterances]
                    _all_files = [f for f in _all_files if f.startswith(utterances)]
                else:
                    _all_files = np.random.permutation(_all_files)
                    _all_files = _all_files[:n_file]
            shouldsave = True
            normconst = None

        _all_files = [os.path.join(self._PATH, f) for f in _all_files]
        self._all_files = [f for f in _all_files if os.path.isfile(f)]

        if norm_class:
            norm_class = eval(f'normalize.{norm_class}')
            if not shouldsave:
                self._normconst = norm_class.load_from_dict(normconst)
            else:
                self._normconst = norm_class.create(
                    self._all_files, cfg.IV_DATA_NAME['x'], cfg.IV_DATA_NAME['y']
                )
        else:
            self._normconst = None

        if shouldsave:
            basenames = [os.path.basename(f) for f in self._all_files]
            dd.io.save(
                f_normconst,
                (basenames,
                 self._normconst.save_to_dict() if self._normconst else None,
                 )
            )
            scio.savemat(
                f_normconst.replace('.h5', '.mat'),
                dict(all_files=basenames,
                     **(self._normconst.save_to_dict(only_consts=True)
                        if self._normconst else dict()
                        )
                     )
            )

        print(self._normconst)
        print(f'{len(self._all_files)} files prepared from {kind_data.upper()}.')

    def __len__(self):
        return len(self._all_files)

    def __getitem__(self, idx: int):
        sample = dict()
        for k, v in self._needs.items():
            if v:
                data = dd.io.load(self._all_files[idx],
                                  group=cfg.IV_DATA_NAME[k],
                                  sel=cfg.CH_SLICES[v])
                if type(data) == np.str_:
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
            if type(value) == str:
                list_data = [batch[idx][key] for idx in idxs_sorted]
                set_data = set(list_data)
                if len(set_data) == 1:
                    result[key] = set_data.pop()
                else:
                    result[key] = list_data
            else:
                # B, T, F, C
                data = [torch.from_numpy(batch[idx][key]).permute(-2, -3, -1)
                        for idx in idxs_sorted]
                data = pad_sequence(data, batch_first=True)
                # B, F, T, C
                data = data.permute(0, -2, -3, -1).numpy()

                result[key] = data

        return result

    @staticmethod
    def decollate_padded(batch: Dict, idx: int) -> Dict:
        result = dict()
        for key, value in batch.items():
            if type(value) == str:
                result[key] = value
            elif type(value) == list:
                result[key] = value[key]
            elif not key.startswith('T_'):
                T_xy = 'T_xs' if 'x' in key else 'T_ys'
                result[key] = value[idx, :, :batch[T_xy][idx], :]
        return result

    @staticmethod
    def save_iv(fname, **kwargs):

        scio.savemat(fname,
                     {cfg.IV_DATA_NAME[k][1:]: v
                      for k, v in kwargs.items() if k in cfg.IV_DATA_NAME}
                     )

    def normalize_on_like(self, other):
        """

        :type other: IVDataset
        :return:
        """
        self._normconst = other._normconst

    def set_needs(self, **kwargs):
        """

        :param kwargs: dict(x: str, y: str,
                            x_denorm: str, y_denorm: str,
                            x_phase: str, y_phase: str)
        :return:
        """
        for k in self._needs:
            if k in kwargs:
                assert (kwargs[k] == 'all' or kwargs[k] == 'alpha'
                        or type(kwargs[k]) == bool)
                self._needs[k] = kwargs[k]

    def normalize(self, a: TensArr, xy: str) -> TensArr:
        return self._normconst.normalize(a, xy)

    def normalize_(self, a: TensArr, xy: str) -> TensArr:
        return self._normconst.normalize_(a, xy)

    def denormalize(self, a: TensArr, xy: str) -> TensArr:
        return self._normconst.denormalize(a, xy)

    def denormalize_(self, a: TensArr, xy: str) -> TensArr:
        return self._normconst.denormalize_(a, xy)

    # noinspection PyProtectedMember
    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
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

        return result
