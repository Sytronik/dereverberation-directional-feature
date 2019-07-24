import multiprocessing as mp
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar, Union, Optional

import numpy as np
import scipy.io as scio
import torch
from numpy import ndarray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from generic import DataPerDevice, TensArr
from audio_utils import LogModule
from hparams import Channel, hp

StrOrSeq = TypeVar('StrOrSeq', str, Sequence[str])
TupOrSeq = TypeVar('TupOrSeq', tuple, Sequence[tuple])
DataDict = Dict[str, Any]


def xy_signature(func):
    def wrapper(self, *args, **kwargs):
        assert (len(args) > 0) ^ (len(kwargs) > 0)
        x, y = func(self, *args, **kwargs)
        if x is not None:
            if y is not None:
                return x, y
            else:
                return x
        else:
            return y

    return wrapper


class Normalization:
    """
    Calculating and saving mean/std of all mel spectrogram with respect to time axis,
    applying normalization to the spectrogram
    This is need only when you don't load all the data on the RAM
    """

    @staticmethod
    def _sum(a: ndarray) -> ndarray:
        return a.sum()

    @staticmethod
    def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
        return ((a - mean_a)**2).sum()

    @staticmethod
    def _load_data(fname: Union[str, Path], key: str, queue: mp.Queue) -> None:
        x = np.load(fname, allow_pickle=True)[key]
        queue.put(x)

    @staticmethod
    def _calc_per_data(data,
                       list_func: Sequence[Callable],
                       args: Sequence = None,
                       ) -> Dict[Callable, Any]:
        """ gather return values of functions in `list_func`

        :param list_func:
        :param args:
        :return:
        """

        if args:
            result = {f: f(data, arg) for f, arg in zip(list_func, args)}
        else:
            result = {f: f(data) for f in list_func}
        return result

    def __init__(self, mean, std):
        self.mean = DataPerDevice(mean)
        self.std = DataPerDevice(std)

    @classmethod
    def calc_const(cls, all_files: List[Path], key: str):
        """

        :param all_files:
        :param key: data name in npz file
        :rtype: Normalization
        """

        # Calculate summation & size (parallel)
        list_fn = (np.size, cls._sum)
        pool_loader = mp.Pool(2)
        pool_calc = mp.Pool(min(mp.cpu_count() - 2, 6))
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, key, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, list_fn)
                ))

        result: List[Dict] = [item.get() for item in result]
        print()

        sum_size = np.sum([item[np.size] for item in result])
        sum_ = np.sum([item[cls._sum] for item in result], axis=0)
        mean = sum_ / (sum_size // sum_.size)

        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, key, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, (cls._sq_dev,), (mean,))
                ))

        pool_loader.close()
        pool_calc.close()
        result: List[Dict] = [item.get() for item in result]
        print()

        sum_sq_dev = np.sum([item[cls._sq_dev] for item in result], axis=0)

        std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)
        print('Calculated Std')

        return cls(mean, std)

    def astuple(self):
        return self.mean.data[ndarray], self.std.data[ndarray]

    # normalize and denormalize functions can accept a ndarray or a tensor.
    def normalize(self, a: TensArr) -> TensArr:
        return (a - self.mean.get_like(a)) / (2 * self.std.get_like(a))

    def normalize_(self, a: TensArr) -> TensArr:  # in-place version
        a -= self.mean.get_like(a)
        a /= 2 * self.std.get_like(a)

        return a

    def denormalize(self, a: TensArr) -> TensArr:
        return a * (2 * self.std.get_like(a)) + self.mean.get_like(a)

    def denormalize_(self, a: TensArr) -> TensArr:  # in-place version
        a *= 2 * self.std.get_like(a)
        a += self.mean.get_like(a)

        return a

    def __str__(self):
        return f'mean - {self.mean[ndarray].shape}, std - {self.std[ndarray].shape}'


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
                 norm_x: Normalization = None, norm_y: Normalization = None,
                 **kwargs: Channel):
        self._PATH = hp.dict_path[f'feature_{kind_data}']

        # default needs
        self._needs = dict(x=Channel.ALL, y=Channel.LAST,
                           x_phase=Channel.NONE, y_phase=Channel.NONE,
                           speech_fname=Channel.ALL)
        self.set_needs(**kwargs)

        self._all_files = [f for f in self._PATH.glob('*.*') if hp.is_featurefile(f)]
        self._all_files = sorted(self._all_files)

        # path_normconst: path of the file that has information about mean, std, ...
        path_normconst = hp.dict_path[f'form_normconst_{kind_data}']

        if kind_data == 'train':
            if not hp.refresh_const or path_normconst.exists():
                # when normconst file exists
                dict_normconst = scio.loadmat(path_normconst, squeeze_me=True)
                self.norm_x = Normalization(*dict_normconst['normconst_x'])
                self.norm_y = Normalization(*dict_normconst['normconst_y'])
            else:
                self.norm_x \
                    = Normalization.calc_const(self._all_files, key=hp.spec_data_names['x'])
                self.norm_y \
                    = Normalization.calc_const(self._all_files, key=hp.spec_data_names['y'])

            print(f'Normalization for input: {self.norm_x}')
            print(f'Normalization for output: {self.norm_y}')
        else:
            assert norm_x, norm_y
            self.norm_x = norm_x
            self.norm_y = norm_y

        print(f'{len(self._all_files)} files are prepared from {kind_data.upper()}.')

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

    def set_needs(self, **kwargs: Channel):
        """ set which data are needed.

        :param kwargs: available keywords are [x, y, x_phase, y_phase, speech_fname]
        """
        for k in self._needs:
            if k in kwargs:
                self._needs[k] = kwargs[k]

    @xy_signature
    def normalize(self, x: TensArr, y: TensArr) -> Union[Optional[TensArr], Tuple]:
        """

        :param x: input dirspec
        :param y: output dirspec
        :return:
        """

        if x is not None:
            x_norm = LogModule.log(x)
            x_norm = self.norm_x.normalize_(x_norm)
        else:
            x_norm = None
        if y is not None:
            y_norm = LogModule.log(y)
            y_norm = self.norm_y.normalize_(y_norm)
        else:
            y_norm = None
        return x_norm, y_norm

    @xy_signature
    def normalize_(self, x: TensArr, y: TensArr) -> Union[Optional[TensArr], Tuple]:
        """

        :param x: input dirspec
        :param y: output dirspec
        :return:
        """

        if x is not None:
            x = LogModule.log_(x)
            x = self.norm_x.normalize_(x)
        if y is not None:
            y = LogModule.log_(y)
            y = self.norm_y.normalize_(y)
        return x, y

    @xy_signature
    def denormalize(self, x: TensArr, y: TensArr) -> Union[Optional[TensArr], Tuple]:
        """

        :param x: input dirspec
        :param y: output dirspec
        :return:
        """

        if x is not None:
            x_denorm = self.norm_x.denormalize(x)
            x_denorm = LogModule.exp_(x_denorm)
        else:
            x_denorm = None
        if y is not None:
            y_denorm = self.norm_y.denormalize(y)
            y_denorm = LogModule.exp_(y_denorm)
        else:
            y_denorm = None
        return x_denorm, y_denorm

    @xy_signature
    def denormalize_(self, x: TensArr, y: TensArr) -> Union[Optional[TensArr], Tuple]:
        """

        :param x: input dirspec
        :param y: output dirspec
        :return:
        """

        if x is not None:
            x = self.norm_x.denormalize_(x)
            x = LogModule.exp_(x)
        if y is not None:
            y = self.norm_y.denormalize_(y)
            y = LogModule.exp_(y)
        return x, y

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

        boundary_i_locs = np.cumsum(
            np.outer(array_n_loc, np.insert(ratio, 0, 0)),
            axis=1, dtype=np.int
        )  # number of rooms x (number of sets + 1)
        boundary_i_locs[:, -1] = array_n_loc
        i_set_per_room_loc: Dict[str, ndarray] = dict()

        for i_room, room in enumerate(rooms):
            i_set = np.empty(array_n_loc[i_room], dtype=np.int)
            for i_b in range(boundary_i_locs.shape[1] - 1):
                range_ = np.arange(boundary_i_locs[i_room, i_b], boundary_i_locs[i_room, i_b + 1])
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
