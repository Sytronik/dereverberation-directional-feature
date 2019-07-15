import os
from copy import copy
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypeVar, Union

import deepdish as dd
import numpy as np
import scipy.io as scio
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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

    __slots__ = ('_PATH', '_needs', '_trannorm', '_all_files')

    def __init__(self, kind_data: str,
                 n_file=-1, keys_trannorm: TupOrSeq = (None,),
                 random_by_utterance=False, **kwargs: Channel):
        self._PATH = hp.dict_path[f'dirspec_{kind_data}']

        # default needs
        self._needs = dict(x=Channel.ALL, y=Channel.MAG,
                           x_phase=Channel.NONE, y_phase=Channel.NONE,
                           speech_fname=Channel.ALL)
        self.set_needs(**kwargs)

        # _all_files (local var): basename of file paths
        # self._all_files: full paths
        self._all_files = None
        trannorm: List[TranNormModule] = []

        # f_normconst: The name of the file
        # that has information about data file list, mean, std, ...
        list_f_norm \
            = glob(hp.dict_path[f'form_normconst_{kind_data}'].format(n_file, '*'))
        for key_tn in keys_trannorm:
            s_f_normconst \
                = hp.dict_path[f'form_normconst_{kind_data}'].format(n_file, key_tn)

            should_calc_save = False
            if s_f_normconst in list_f_norm:
                # when normconst file exists
                s_all_files, normconst = dd.io.load(s_f_normconst)
            elif not self._all_files:
                if len(list_f_norm) > 0:
                    # load file list from another normconst file
                    s_all_files, _ = dd.io.load(list_f_norm[0])
                else:
                    # search all data files
                    s_all_files = [
                        f.name for f in os.scandir(self._PATH) if hp.is_featurefile(f)
                    ]
                    s_all_files = sorted(s_all_files)
                    if n_file != -1:
                        if random_by_utterance:
                            utterances = np.random.randint(
                                len(s_all_files) // hp.n_loc[kind_data],
                                size=n_file // hp.n_loc[kind_data]
                            )
                            utterances = [f'{u:4d}_' for u in utterances]
                            s_all_files = [
                                f for f in s_all_files if f.startswith(utterances)
                            ]
                        else:
                            s_all_files = np.random.permutation(s_all_files)
                            s_all_files = s_all_files[:n_file]
                normconst = None
                should_calc_save = True
            else:
                # if already has file list
                s_all_files = [f.name for f in self._all_files]
                normconst = None
                should_calc_save = True

            # full paths of only existing files
            if not self._all_files:
                self._all_files = [self._PATH / f for f in s_all_files]
                self._all_files = [f for f in self._all_files if f.exists()]

            if hp.refresh_const:
                should_calc_save = True

            if key_tn:
                if should_calc_save:
                    trannorm.append(TranNormModule.create_module(key_tn, self._all_files))
                else:
                    trannorm.append(TranNormModule.load_module(key_tn, normconst))

            if trannorm[-1]:
                if should_calc_save:
                    dd.io.save(s_f_normconst, (s_all_files, trannorm[-1].consts))
                scio.savemat(s_f_normconst.replace('.h5', '.mat'),
                             dict(all_files=s_all_files,
                                  **trannorm[-1].consts_as_dict()
                                  )
                             )
            else:
                if should_calc_save:
                    dd.io.save(s_f_normconst, (s_all_files, None))
                scio.savemat(s_f_normconst.replace('.h5', '.mat'),
                             dict(all_files=s_all_files)
                             )

            print(trannorm[-1])

        self._trannorm = trannorm
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
                data = dd.io.load(self._all_files[idx],
                                  group=hp.spec_data_name[k],
                                  **v.value)
                if type(data) == np.str_:
                    sample[k] = str(data)
                else:
                    sample[k] = data.astype(np.float32)

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
                data = data.permute(0, -2, -3, -1)  # .numpy()

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
        scio.savemat(str(fname),
                     {hp.spec_data_name[k][1:]: v
                      for k, v in kwargs.items() if k in hp.spec_data_name}
                     )

    def normalize_on_like(self, other):
        """ use the same TranNormModule as other uses.

        :type other: DirSpecDataset
        :return:
        """
        self._trannorm = other._trannorm

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
        return self._trannorm[idx].process(xy, a, a_phase)

    def preprocess_(self, xy: XY, a: TensArr, a_phase: TensArr = None, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param a_phase: phase spectrum
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorm[idx].process_(xy, a, a_phase)

    def set_transformer_var(self, idx=0, **kwargs):
        self._trannorm[idx].set_transformer_var(**kwargs)

    def postprocess(self, xy: XY, a: TensArr, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorm[idx].inverse(xy, a)

    def postprocess_(self, xy: XY, a: TensArr, idx=0) -> TensArr:
        """

        :param xy: one of 'x' or 'y'
        :param a: directional spectrogram
        :param idx: index of self._trannorm
        :return:
        """
        return self._trannorm[idx].inverse_(xy, a)

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
