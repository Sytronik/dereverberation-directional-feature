from itertools import islice
from typing import Callable, Dict, List, Sequence, Tuple, Type

import deepdish as dd
import numpy as np
import torch
from torch import Tensor

import config as cfg
from generic import TensArr, TensArrOrSeq
from .normalize import INormalizer, MeanStdNormalizer, MinMaxNormalizer
from .transform import (ITransformer,
                        LogMagTransformer, LogReImTransformer,
                        MagTransformer, ReImTransformer,
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
    }

    @classmethod
    def create_module(cls, key: Tuple[str, str, bool], all_files: List[str], kwargs_normalizer=None):
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
        module = cls(*cls.DICT_MODULE_ARGS[key])
        module.consts = consts

        return module

    def __init__(self, transformer: Type[ITransformer], normalizer: Type[INormalizer]):
        super().__init__()
        self._transformer = transformer
        self._normalizer = normalizer

        self.consts: Tuple[TensArr] = None
        self.__cpu: Tuple[Tensor] = None
        self.__cuda: Tuple[Tensor] = dict()

    def _calc_per_file(self, fname: str, list_func: Sequence[Callable],
                       args_x: Sequence = None, args_y: Sequence = None) -> Tuple:
        while True:
            try:
                if self._transformer.USE_PHASE:
                    x, x_phase, y, y_phase \
                        = dd.io.load(fname, group=(cfg.IV_DATA_NAME['x'],
                                                   cfg.IV_DATA_NAME['x_phase'],
                                                   cfg.IV_DATA_NAME['y'],
                                                   cfg.IV_DATA_NAME['y_phase'],
                                                   ))
                else:
                    x, y = dd.io.load(fname, group=(cfg.IV_DATA_NAME['x'],
                                                    cfg.IV_DATA_NAME['y'],
                                                    ))
                    x_phase = None
                    y_phase = None
                break
            except:  # noqa: E722
                print(fname)
                fname_sq = fname.replace('IV_sqrt', 'IV')
                iv_dict = dd.io.load(fname_sq)
                iv_dict['IV_free'][..., -1] = np.sqrt(iv_dict['IV_free'][..., -1])
                iv_dict['IV_room'][..., -1] = np.sqrt(iv_dict['IV_room'][..., -1])
                dd.io.save(fname, iv_dict, compression=None)

        x = self._transformer.transform_(x, x_phase)
        y = self._transformer.transform_(y, y_phase)

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
    def _tensor_seq(seq: Sequence[TensArr], device=torch.device('cpu')):
        return [torch.tensor(a, device=device) for a in seq]

    def _get_consts_like(self, xy: str, a: TensArr):
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

    def consts_as_dict(self) -> Dict:
        # return {k: a for k, a in self.__dict__.items()
        #         if not k.startswith('_')}
        xy = ['x', 'y']
        keys = [f'{self._normalizer.names[i//2]}_{xy[i%2]}' for i in range(len(self.consts))]
        return {key: const.squeeze() for key, const in zip(keys, self.consts)}

    def __len__(self):
        return len(self.consts)

    def __str__(self):
        result = (f'{self._transformer.__class__.__name__} & '
                  f'{self._normalizer.__class__.__name__}\n')
        for idx, name in enumerate(self._normalizer.names):
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
        consts = self._get_consts_like(xy, b)

        if a is b:
            return self._normalizer.normalize(b, *consts)
        else:
            return self._normalizer.normalize_(b, *consts)

    def process_(self, xy: str, a: TensArr, a_phase: TensArr) -> TensArr:
        a = self._transformer.transform_(a, a_phase)
        consts = self._get_consts_like(xy, a)

        return self._normalizer.normalize_(a, *consts)

    def inverse(self, xy: str, a: TensArr) -> TensArrOrSeq:
        consts = self._get_consts_like(xy, a)
        tup = self._transformer.inverse_(self._normalizer.denormalize(a, *consts))
        if tup[1] is None:
            return tup[0]
        else:
            return tup

    def inverse_(self, xy: str, a: TensArr) -> TensArrOrSeq:
        consts = self._get_consts_like(xy, a)
        tup = self._transformer.inverse_(self._normalizer.denormalize_(a, *consts))
        if tup[1] is None:
            return tup[0]
        else:
            return tup