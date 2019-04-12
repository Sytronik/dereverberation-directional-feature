import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from generic import TensArr, TensArrOrSeq


class INormalizer(metaclass=ABCMeta):
    """ Normalizer Interface """

    @staticmethod
    @abstractmethod
    def names() -> tuple:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def calc_consts(cls, fn_load_data: Callable, fn_calc_per_data: Callable, all_files: List[str],
                    **kwargs) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def normalize(a: TensArr, *consts) -> TensArr:
        pass

    @staticmethod
    @abstractmethod
    def normalize_(a: TensArr, *consts) -> TensArr:
        pass

    @staticmethod
    @abstractmethod
    def denormalize(a: TensArr, *consts) -> TensArrOrSeq:
        pass

    @staticmethod
    @abstractmethod
    def denormalize_(a: TensArr, *consts) -> TensArrOrSeq:
        pass


class MeanStdNormalizer(INormalizer):
    @staticmethod
    def names() -> tuple:
        return 'mean', 'std'

    @staticmethod
    def _sum(a: ndarray) -> ndarray:
        return a.sum(axis=1, keepdims=True)

    @staticmethod
    def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
        return ((a - mean_a)**2).sum(axis=1, keepdims=True)

    @classmethod
    def calc_consts(cls, fn_load_data: Callable, fn_calc_per_data: Callable, all_files: List[Path],
                    **kwargs) -> tuple:
        need_mean = kwargs.get('need_mean', True)

        # Calculate summation & size (parallel)
        list_fn = (np.size, cls._sum) if need_mean else (np.size,)
        pool_loader = mp.Pool(2)
        pool_calc = mp.Pool(mp.cpu_count()-4)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(fn_load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = [None] * len(all_files)
            for idx in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result[idx] = pool_calc.apply_async(
                    fn_calc_per_data,
                    (*data, list_fn)
                )

        result: Tuple[dict] = [item.get() for item in result]
        print()

        sum_size_x = np.sum([item[0][np.size] for item in result])
        sum_size_y = np.sum([item[1][np.size] for item in result])
        if need_mean:
            sum_x = np.sum([item[0][cls._sum] for item in result], axis=0)
            sum_y = np.sum([item[1][cls._sum] for item in result], axis=0)
            mean_x = sum_x / (sum_size_x // sum_x.size)
            mean_y = sum_y / (sum_size_y // sum_y.size)
            # mean_x = sum_x[..., :3] / (sum_size_x//sum_x[..., :3].size)
            # mean_y = sum_y[..., :3] / (sum_size_y//sum_y[..., :3].size)
        else:
            mean_x = 0.
            mean_y = 0.
        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(fn_load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = [None] * len(all_files)
            for idx in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result[idx] = pool_calc.apply_async(
                    fn_calc_per_data,
                    (*data, (cls._sq_dev,), (mean_x,), (mean_y,))
                )

        pool_loader.close()
        pool_calc.close()
        result: Tuple[dict] = [item.get() for item in result]
        print()

        sum_sq_dev_x = np.sum([item[0][cls._sq_dev] for item in result], axis=0)
        sum_sq_dev_y = np.sum([item[1][cls._sq_dev] for item in result], axis=0)

        std_x = np.sqrt(sum_sq_dev_x / (sum_size_x // sum_sq_dev_x.size) + 1e-5)
        std_y = np.sqrt(sum_sq_dev_y / (sum_size_y // sum_sq_dev_y.size) + 1e-5)
        print('Calculated Std')

        return mean_x, mean_y, std_x, std_y

    @staticmethod
    def normalize(a: TensArr, *consts) -> TensArr:
        return (a - consts[0]) / (2 * consts[1])

    @staticmethod
    def normalize_(a: TensArr, *consts) -> TensArr:
        a -= consts[0]
        a /= 2 * consts[1]

        return a

    @staticmethod
    def denormalize(a: TensArr, *consts) -> TensArrOrSeq:
        return a * (2 * consts[1]) + consts[0]

    @staticmethod
    def denormalize_(a: TensArr, *consts) -> TensArrOrSeq:
        a *= 2 * consts[1]
        a += consts[0]

        return a


class MinMaxNormalizer(INormalizer):
    @staticmethod
    def names() -> tuple:
        return 'min', 'max'

    @staticmethod
    def _min(a: ndarray) -> ndarray:
        return a.min(axis=1, keepdims=True)

    @staticmethod
    def _max(a: ndarray) -> ndarray:
        return a.max(axis=1, keepdims=True)

    @classmethod
    def calc_consts(cls, fn_load_data: Callable, fn_calc_per_data: Callable, all_files: List[str],
                    **kwargs) -> tuple:
        # Calculate summation & no. of total frames (parallel)
        pool_loader = mp.Pool(3)
        pool = mp.Pool(mp.cpu_count())
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(fn_load_data,
                                      [(f, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = [None] * len(all_files)
            for idx in range(len(all_files)):
                data = queue_data.get()
                result[idx] = pool.apply_async(
                    fn_calc_per_data,
                    (*data, (cls._min, cls._max),)
                )
        pool.close()
        result: Tuple[dict] = [item.get() for item in result]

        min_x = np.min([res[0][cls._min] for res in result], axis=0)
        min_y = np.min([res[1][cls._min] for res in result], axis=0)
        max_x = np.max([res[0][cls._max] for res in result], axis=0)
        max_y = np.max([res[1][cls._max] for res in result], axis=0)
        print('Calculated Min/Max.')

        return min_x, min_y, max_x, max_y

    @staticmethod
    def normalize(a: TensArr, *consts) -> TensArr:
        return (a - consts[0]) / (consts[1] - consts[0]) - 0.5

    @staticmethod
    def normalize_(a: TensArr, *consts) -> TensArr:
        a -= consts[0]
        a /= (consts[1] - consts[0])
        a -= 0.5

        return a

    @staticmethod
    def denormalize(a: TensArr, *consts) -> TensArrOrSeq:
        return (a + 0.5) * (consts[1] - consts[0]) + consts[0]

    @staticmethod
    def denormalize_(a: TensArr, *consts) -> TensArrOrSeq:
        a += 0.5
        a *= (consts[1] - consts[0])
        a += consts[0]
        return a
