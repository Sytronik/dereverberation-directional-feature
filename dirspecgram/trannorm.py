from enum import IntEnum
from itertools import islice
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, Union, ClassVar

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from hparams import hp
from generic import TensArr, TensArrOrSeq
from .normalize import INormalizer, MeanStdNormalizer, MinMaxNormalizer
from .transform import (ITransformer,
                        LogMagTransformer, LogReImTransformer,
                        MagTransformer, ReImTransformer,
                        LogMagBPDTransformer,
                        )


class XY(IntEnum):
    x = 0
    y = 1


class TranNormModule:
    __slots__ = 'consts', '__cpu', '__cuda', '_transformer', '_normalizer'

    DICT_MODULE_ARGS: ClassVar[Dict[tuple, tuple]] = {
        ('mag', 'meanstd', True): (LogMagTransformer, MeanStdNormalizer),
        ('mag', 'meanstd', False): (MagTransformer, MeanStdNormalizer),
        ('mag', 'minmax', True): (LogMagTransformer, MinMaxNormalizer),
        ('mag', 'minmax', False): (MagTransformer, MinMaxNormalizer),
        ('complex', 'meanstd', True): (LogReImTransformer, MeanStdNormalizer),
        ('complex', 'meanstd', False): (ReImTransformer, MeanStdNormalizer),
        ('magbpd', 'meanstd', True): (LogMagBPDTransformer, MeanStdNormalizer),
        # ('magbpd', 'meanstd', False): (MagBPDTransformer, MeanStdNormalizer),
    }

    @classmethod
    def create_module(cls, key: Tuple[str, str, bool], all_files: List[str],
                      kwargs_normalizer=None):
        """ create module by calculating normalize constants.

        :param key:
        :param all_files:
        :param kwargs_normalizer:
        :rtype: TranNormModule
        """
        if not kwargs_normalizer:
            kwargs_normalizer = dict()

        module = cls(*cls.DICT_MODULE_ARGS[key])

        consts = list(module._normalizer.calc_consts(module._load_data, module._calc_per_data,
                                                     all_files,
                                                     **kwargs_normalizer))
        if consts[0].dtype == np.float64:
            consts = [a.astype(np.float32) for a in consts]
        module.consts = consts

        return module

    @classmethod
    def load_module(cls, key: Tuple[str, str, bool], consts: Tuple[ndarray]):
        """ create module with `consts` which is normalize constants.

        :param key:
        :param consts:
        :rtype: TranNormModule
        """
        module = cls(*cls.DICT_MODULE_ARGS[key])
        module.consts = consts

        return module

    def __init__(self, transformer: Type[ITransformer], normalizer: Type[INormalizer]):
        super().__init__()
        self._transformer = transformer
        self._normalizer = normalizer

        self.consts: Sequence[TensArr] = None
        self.__cpu: Sequence[Tensor] = None
        self.__cuda: Dict[torch.device, Sequence[Tensor]] = dict()

    def _load_data(self, fname: Union[str, Path], queue: Queue) -> None:
        npz = np.load(fname)
        if self._transformer.use_phase():
            result = [npz[hp.spec_data_names[name]] for name in ('x', 'x_phase', 'y', 'y_phase')]
            queue.put(result)
        else:
            x, y = [npz[hp.spec_data_names[name]] for name in ('x', 'y')]
            queue.put((x, y, None, None))

    def _calc_per_data(self, x, y, x_phase, y_phase,
                       list_func: Sequence[Callable],
                       args_x: Sequence = None,
                       args_y: Sequence = None,
                       ) -> Tuple[Dict[Callable, Any], Dict[Callable, Any]]:
        """ gather return values of functions in `list_func`
         with data in a file `fname`

        :param fname:
        :param list_func:
        :param args_x:
        :param args_y:
        :return:
        """

        x = self._transformer.transform_(x, x_phase)
        y = self._transformer.transform_(y, y_phase)
        channels_x = [i for i in np.arange(-x.shape[-1], 0)
                      if i not in self._transformer.no_norm_channels]
        channels_y = [i for i in np.arange(-y.shape[-1], 0)
                      if i not in self._transformer.no_norm_channels]

        x = x[..., channels_x]
        y = y[..., channels_y]

        if args_x:
            result_x = {f: f(x, arg) for f, arg in zip(list_func, args_x)}
        else:
            result_x = {f: f(x) for f in list_func}
        if args_y:
            result_y = {f: f(y, arg) for f, arg in zip(list_func, args_y)}
        else:
            result_y = {f: f(y) for f in list_func}

        # print('.', end='', flush=True)
        return result_x, result_y

    @staticmethod
    def _tensor_seq(seq: Sequence[TensArr], device=torch.device('cpu')) \
            -> Sequence[Tensor]:
        return [torch.tensor(a, device=device) for a in seq]

    def _get_consts_like(self, xy: XY, a: TensArr) -> List[TensArr]:
        """ get consts for `xy` in the type of `a`
         (if `a` is a Ttensor, in the same device as `a`)

        :param xy:
        :param a:
        :return:
        """

        shft = int(xy)

        ch = slice(-a.shape[-1], None)

        if type(a) == torch.Tensor:
            if a.device == torch.device('cpu'):
                if not self.__cpu:
                    self.__cpu = self._tensor_seq(self.consts, device=a.device)
                result = self.__cpu
            else:
                if a.device not in self.__cuda:
                    self.__cuda[a.device] = self._tensor_seq(self.consts, device=a.device)
                result = self.__cuda[a.device]
        else:
            result = self.consts

        return [c[..., ch] if c.shape else c for c in islice(result, shft, None, 2)]

    def consts_as_dict(self) -> Dict[str, ndarray]:
        xy = ('x', 'y')
        keys = [f'{self._normalizer.names()[i // 2]}_{xy[i % 2]}'
                for i in range(len(self.consts))]
        return {key: const.squeeze() for key, const in zip(keys, self.consts)}

    def __len__(self):
        return len(self.consts)

    def __str__(self):
        result = ['']*(len(self._normalizer.names())+1)
        result[0] = f'{self._transformer.__name__} & {self._normalizer.__name__}'
        for idx, name in enumerate(self._normalizer.names()):
            x = self.consts[2 * idx]
            y = self.consts[2 * idx + 1]

            result[idx+1] = f'{name}: '
            if hasattr(x, 'shape') and x.shape:
                result[idx+1] += f'{x.shape}, {y.shape}\t'
            else:
                result[idx+1] += f'{x}, {y}\t'

        return '\n'.join(result)

    def process(self, xy: XY, a: TensArr, a_phase: TensArr) -> TensArr:
        b = self._transformer.transform(a, a_phase)
        if self._transformer.no_norm_channels:
            channels = [i for i in range(-b.shape[-1], 0)
                        if i not in self._transformer.no_norm_channels]
            if a is b:
                b = a[:]
            b_norm = b[..., channels]
            consts = self._get_consts_like(xy, b_norm)
            b[..., channels] = self._normalizer.normalize_(b_norm, *consts)

            return b
        else:
            consts = self._get_consts_like(xy, b)

            if a is b:
                return self._normalizer.normalize(b, *consts)
            else:
                return self._normalizer.normalize_(b, *consts)

    def process_(self, xy: XY, a: TensArr, a_phase: TensArr) -> TensArr:
        a = self._transformer.transform_(a, a_phase)
        if self._transformer.no_norm_channels:
            channels = [i for i in range(-a.shape[-1], 0)
                        if i not in self._transformer.no_norm_channels]
            a_norm = a[..., channels]
            consts = self._get_consts_like(xy, a_norm)
            a[..., channels] = self._normalizer.normalize_(a_norm, *consts)

            return a
        else:
            consts = self._get_consts_like(xy, a)

            return self._normalizer.normalize_(a, *consts)

    def set_transformer_var(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self._transformer, k):
                setattr(self._transformer, k, v)

    def inverse(self, xy: XY, a: TensArr) -> TensArrOrSeq:
        if self._transformer.no_norm_channels:
            channels = [i for i in range(-a.shape[-1], 0)
                        if i not in self._transformer.no_norm_channels]
            a_norm = a[..., channels]
            consts = self._get_consts_like(xy, a_norm)
            b = a[:]
            b[..., channels] = self._normalizer.denormalize(a_norm, *consts)
            tup = self._transformer.inverse_(b)
        else:
            consts = self._get_consts_like(xy, a)
            tup = self._transformer.inverse_(self._normalizer.denormalize(a, *consts))

        return tup[0] if tup[1] is None else tup

    def inverse_(self, xy: XY, a: TensArr) -> TensArrOrSeq:
        if self._transformer.no_norm_channels:
            channels = [i for i in range(-a.shape[-1], 0)
                        if i not in self._transformer.no_norm_channels]
            a_norm = a[..., channels]
            consts = self._get_consts_like(xy, a_norm)
            a[..., channels] = self._normalizer.denormalize(a_norm, *consts)
            tup = self._transformer.inverse_(a)
        else:
            consts = self._get_consts_like(xy, a)
            tup = self._transformer.inverse_(self._normalizer.denormalize_(a, *consts))

        return tup[0] if tup[1] is None else tup
