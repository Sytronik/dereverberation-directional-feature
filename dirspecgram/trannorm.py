from itertools import islice
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type

import deepdish as dd
import numpy as np
import torch
from torch import Tensor
from numpy import ndarray

import config as cfg
import generic as gen
from generic import TensArr, TensArrOrSeq
from .normalize import INormalizer, MeanStdNormalizer, MinMaxNormalizer
from .transform import (ITransformer,
                        LogMagTransformer, LogReImTransformer,
                        MagTransformer, ReImTransformer,
                        LogMagBPDTransformer
                        )


class TranNormModule:
    __slots__ = 'consts', '__cpu', '__cuda', '_transformer', '_normalizer'

    DICT_MODULE_ARGS = {
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

        consts = list(module._normalizer.calc_consts(module._calc_per_file,
                                                     all_files,
                                                     **kwargs_normalizer))
        if consts[0].dtype == np.float64:
            consts = [a.astype(np.float32) for a in consts]
        module.consts = consts

        return module

    @classmethod
    def load_module(cls, key: Tuple[str, str, bool], consts: Tuple[TensArr]):
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

    def _calc_per_file(self, fname: str,
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

        if self._transformer.use_phase():
            x, x_phase, y, y_phase \
                = dd.io.load(fname, group=(cfg.SPEC_DATA_NAME['x'],
                                           cfg.SPEC_DATA_NAME['x_phase'],
                                           cfg.SPEC_DATA_NAME['y'],
                                           cfg.SPEC_DATA_NAME['y_phase'],
                                           ))
        else:
            x, y = dd.io.load(fname, group=(cfg.SPEC_DATA_NAME['x'],
                                            cfg.SPEC_DATA_NAME['y'],
                                            ))
            x_phase = None
            y_phase = None

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

        print('.', end='', flush=True)
        return result_x, result_y

    @staticmethod
    def _tensor_seq(seq: Sequence[TensArr], device=torch.device('cpu')) \
            -> Sequence[Tensor]:
        return [torch.tensor(a, device=device) for a in seq]

    def _get_consts_like(self, xy: str, a: TensArr) -> List[TensArr]:
        """ get consts for `xy` in the type of `a`
         (if `a` is a Ttensor, in the same device as `a`)

        :param xy:
        :param a:
        :return:
        """

        assert xy == 'x' or xy == 'y'
        shft = int(xy == 'y') if not cfg.NORM_USING_ONLY_X else 0

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
        xy = ['x', 'y']
        keys = [f'{self._normalizer.names()[i // 2]}_{xy[i % 2]}'
                for i in range(len(self.consts))]
        return {key: const.squeeze() for key, const in zip(keys, self.consts)}

    def __len__(self):
        return len(self.consts)

    def __str__(self):
        result = (f'{self._transformer.__name__} & '
                  f'{self._normalizer.__name__}\n')
        for idx, name in enumerate(self._normalizer.names()):
            x = self.consts[2 * idx]
            y = self.consts[2 * idx + 1]

            result += f'{name}: '
            if hasattr(x, 'shape') and x.shape:
                result += f'{x.shape}, {y.shape}\t'
            else:
                result += f'{x}, {y}\t'

        return result[:-1]

    def process(self, xy: str, a: TensArr, a_phase: TensArr) -> TensArr:
        b = self._transformer.transform(a, a_phase)
        if self._transformer.no_norm_channels:
            channels = [i for i in np.arange(-b.shape[-1], 0)
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

    def process_(self, xy: str, a: TensArr, a_phase: TensArr) -> TensArr:
        a = self._transformer.transform_(a, a_phase)
        if self._transformer.no_norm_channels:
            channels = [i for i in np.arange(-a.shape[-1], 0)
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

    def inverse(self, xy: str, a: TensArr) -> TensArrOrSeq:
        if self._transformer.no_norm_channels:
            channels = [i for i in np.arange(-a.shape[-1], 0)
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

    def inverse_(self, xy: str, a: TensArr) -> TensArrOrSeq:
        if self._transformer.no_norm_channels:
            channels = [i for i in np.arange(-a.shape[-1], 0)
                        if i not in self._transformer.no_norm_channels]
            a_norm = a[..., channels]
            consts = self._get_consts_like(xy, a_norm)
            a[..., channels] = self._normalizer.denormalize(a_norm, *consts)
            tup = self._transformer.inverse_(a)
        else:
            consts = self._get_consts_like(xy, a)
            tup = self._transformer.inverse_(self._normalizer.denormalize_(a, *consts))

        return tup[0] if tup[1] is None else tup
