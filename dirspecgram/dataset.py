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
        with np.load(self._all_files[idx], mmap_mode='r') as npz_data:
            for k, v in self._needs.items():
                if v.value:
                    data: ndarray = npz_data[hp.spec_data_names[k]]
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

        # TODO not use n_loc, use filename convention instead
        metadata = scio.loadmat(dataset._PATH / 'metadata.mat',
                                squeeze_me=True,
                                chars_as_strings=True)
        array_n_loc = metadata['n_loc']
        rooms = [r.rstrip() for r in metadata['rooms']]

        boundaries = np.cumsum(
            np.outer(array_n_loc, np.insert(ratio, 0, 0)),
            axis=1, dtype=np.int
        )  # number of rooms x (number of sets + 1)
        boundaries[:, -1] = array_n_loc
        i_set_per_room_loc: Dict[str, ndarray] = dict()

        for i_room, room in enumerate(rooms):
            i_set = np.empty(array_n_loc[i_room], dtype=np.int)
            for i_b in range(boundaries.shape[1] - 1):
                range_ = np.arange(boundaries[i_room, i_b], boundaries[i_room, i_b+1])
                i_set[range_] = i_b
            i_set_per_room_loc[room] = i_set

        all_files = dataset._all_files
        dataset._all_files = None
        result = [copy(dataset) for _ in range(n_split)]
        for set_ in result:
            set_._all_files = []
        for f in all_files:
            _, _, room, i_loc = f.stem.split('_')
            i_loc = int(i_loc)
            result[i_set_per_room_loc[room][i_loc]]._all_files.append(f)

        return result
