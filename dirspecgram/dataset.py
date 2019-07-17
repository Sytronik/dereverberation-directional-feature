import os
from copy import copy
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypeVar, Union, Tuple

import numpy as np
import scipy.io as scio
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from numpy import ndarray

from hparams import hp, Channel
from generic import TensArr
from .trannorm import TranNormModule, XY

StrOrSeq = TypeVar('StrOrSeq', str, Sequence[str])
TupOrSeq = TypeVar('TupOrSeq', tuple, Sequence[tuple])
DataDict = Dict[str, Any]


class DirSpecDataset(Dataset):
    """ Directional Spectrogram Dataset

    In the `split` class method, following member variables will be copied or split.
    (will be copied)
    _PATH
    _needs
    _trannorm

    (will be split)
    _all_files
    """

    def __init__(self, kind_data: str,
                 keys_trannorm: TupOrSeq = (None,), **kwargs: Channel):
        self._PATH = hp.dict_path[f'feature_{kind_data}']

        # default needs
        self._needs = dict(x=Channel.ALL, y=Channel.MAG,
                           x_phase=Channel.NONE, y_phase=Channel.NONE,
                           speech_fname=Channel.ALL)
        self.set_needs(**kwargs)

        self._all_files = [f for f in self._PATH.glob('*.*') if hp.is_featurefile(f)]
        self._all_files = sorted(self._all_files)
        self.n_loc = hp.n_loc[kind_data]

        trannorms: List[TranNormModule] = []

        # f_normconst: The name of the file
        # that has information about mean, std, ...
        list_f_norm \
            = glob(hp.dict_path[f'form_normconst_{kind_data}'].format('*'))
        for key_tn in keys_trannorm:
            if not key_tn:
                continue
            s_f_normconst \
                = hp.dict_path[f'form_normconst_{kind_data}'].format(key_tn)

            if not hp.refresh_const and s_f_normconst in list_f_norm:
                # when normconst file exists
                dict_consts: Dict[str, ndarray] = dict(**np.load(s_f_normconst))
                normconst = tuple(dict_consts.values())
                trannorms.append(TranNormModule.load_module(key_tn, normconst))
            else:
                trannorms.append(TranNormModule.create_module(key_tn, self._all_files))
                np.savez(s_f_normconst, *trannorms[-1].consts)
                scio.savemat(s_f_normconst.replace('.npz', '.mat'),
                             trannorms[-1].consts_as_dict())

            print(trannorms[-1])

        self._trannorms = trannorms
        print(f'{len(self._all_files)} files prepared from {kind_data.upper()}.')

    def __len__(self):
        return len(self._all_files)

    def __getitem__(self, idx: int) -> DataDict:
        """

        :param idx:
        :return: DataDict
            Values can be an integer, ndarray, or str.
        """
        sample = dict()
        for k, v in self._needs.items():
            if v.value:
                data: ndarray = np.load(self._all_files[idx])[hp.spec_data_names[k]]
                data = data[..., v.value]
                if type(data) == np.str_:
                    sample[k] = str(data)
                else:
                    sample[k] = torch.from_numpy(data.astype(np.float32))

        for xy in ('x', 'y'):
            sample[f'T_{xy}'] = sample[xy].shape[-2]

        return sample

    @staticmethod
    @torch.no_grad()
    def pad_collate(batch: List[DataDict]) -> DataDict:
        """ return data with zero-padding

        Important data like x, y are all converted to Tensor(cpu).
        :param batch:
        :return: DataDict
            Values can be an Tensor(cpu), list of str, ndarray of int.
        """
        result = dict()
        T_xs = np.array([item.pop('T_x') for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted].tolist()
        T_ys = [batch[idx].pop('T_y') for idx in idxs_sorted]

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
                data = [batch[idx][key].permute(-2, -3, -1) for idx in idxs_sorted]
                data = pad_sequence(data, batch_first=True)
                # B, F, T, C
                data = data.permute(0, -2, -3, -1).contiguous()

                result[key] = data

        return result

    @staticmethod
    @torch.no_grad()
    def decollate_padded(batch: DataDict, idx: int) -> DataDict:
        """ select the `idx`-th data, get rid of padded zeros and return it.

        Important data like x, y are all converted to ndarray.
        :param batch:
        :param idx:
        :return: DataDict
            Values can be an str or ndarray.
        """
        result = dict()
        for key, value in batch.items():
            if type(value) == str:
                result[key] = value
            elif type(value) == list:
                result[key] = value[idx]
            elif not key.startswith('T_'):
                T_xy = 'T_xs' if 'x' in key else 'T_ys'
                result[key] = value[idx, :, :batch[T_xy][idx], :].numpy()
        return result

    @staticmethod
    def save_dirspec(fname: Union[str, Path], **kwargs):
        """ save directional spectrograms.

        :param fname:
        :param kwargs:
        :return:
        """
        scio.savemat(fname,
                     {hp.spec_data_names[k]: v
                      for k, v in kwargs.items() if k in hp.spec_data_names}
                     )

    def normalize_on_like(self, other):
        """ use the same TranNormModule as other uses.

        :type other: DirSpecDataset
        :return:
        """
        self._trannorms = other._trannorms

    def set_needs(self, **kwargs: Channel):
        """ set which data are needed.

        :param kwargs: available keywords are [x, y, x_phase, y_phase, speech_fname]
        """
        for k in self._needs:
            if k in kwargs:
                self._needs[k] = kwargs[k]

    def preprocess(self, xy: XY, a: TensArr, a_phase: TensArr = None, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param a_phase: phase spectrum
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorms[idx].process(xy, a, a_phase)

    def preprocess_(self, xy: XY, a: TensArr, a_phase: TensArr = None, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param a_phase: phase spectrum
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorms[idx].process_(xy, a, a_phase)

    def set_transformer_var(self, idx=0, **kwargs):
        self._trannorms[idx].set_transformer_var(**kwargs)

    def postprocess(self, xy: XY, a: TensArr, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorms[idx].inverse(xy, a)

    def postprocess_(self, xy: XY, a: TensArr, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorms[idx].inverse_(xy, a)

    # noinspection PyProtectedMember
    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """ Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: DirSpecDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[DirSpecDataset]
        """
        if type(dataset) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[mask] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[mask] = 1 - ratio.sum()

        i_loc_split = np.cumsum(np.insert(ratio, 0, 0) * dataset.n_loc, dtype=int)
        i_loc_split[-1] = dataset.n_loc
        pivot = -1
        i_sets = [0] * dataset.n_loc
        for i_loc in range(dataset.n_loc):
            if i_loc_split[pivot + 1] == i_loc:
                pivot += 1
            i_sets[i_loc] = pivot

        all_files = dataset._all_files
        dataset._all_files = None
        result = [copy(dataset) for _ in range(n_split)]
        for f in all_files:
            i_loc = int(f.stem.split('_')[-1])
            if result[i_sets[i_loc]]._all_files:
                result[i_sets[i_loc]]._all_files.append(f)
            else:
                result[i_sets[i_loc]]._all_files = [f]

        return result
