"""
Generic type & functions for torch.Tensor and np.ndarray
"""

import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

from typing import Tuple, Union, List, TypeVar


TensArr = TypeVar('TensArr', Tensor, ndarray)


def convert(a: TensArr, astype: type) -> TensArr:
    if astype == Tensor:
        if type(a) == Tensor:
            return a
        else:
            return torch.as_tensor(a, dtype=torch.float32)
    elif astype == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(astype)


def ndim(a: TensArr) -> int:
    if type(a) == Tensor:
        return a.dim()
    elif type(a) == ndarray:
        return a.ndim
    else:
        raise TypeError


def transpose(a: TensArr,
              axes: Union[int, Tuple[int, ...], List[int]]=None) -> TensArr:
    if type(a) == Tensor:
        if not axes:
            if a.dim() >= 2:
                return a.permute((1, 0)+(-1,)*(a.dim()-2))
            else:
                return a
        else:
            return a.permute(axes)

    elif type(a) == ndarray:
        if a.ndim == 1 and not axes:
            return a
        else:
            return a.transpose(axes)
    else:
        raise TypeError


def squeeze(a: TensArr, axis=None) -> TensArr:
    if type(a) == Tensor:
        return a.squeeze(dim=axis)
    elif type(a) == ndarray:
        return a.squeeze(axis=axis)
    else:
        raise TypeError


def _cat_stack(fn: str,
               a: Union[Tuple[TensArr, ...], List[TensArr]],
               axis=0,
               astype: type=None) -> TensArr:
    fn_dict = {(torch, 'cat'): torch.cat,
               (np, 'cat'): np.concatenate,
               (torch, 'stack'): torch.stack,
               (np, 'stack'): np.stack,
               }

    types = [type(item) for item in a]
    if np.any(types != types[0]):
        a = [convert(item, (astype if astype else types[0])) for item in a]

    if types[0] == Tensor:
        result = fn_dict[(torch, fn)](a, dim=axis)
    elif types[0] == ndarray:
        result = fn_dict[(np, fn)](a, axis=axis)
    else:
        raise TypeError

    return convert(result, astype) if astype else result


def cat(*args, **kargs) -> TensArr:
    """
    <parameters>
    a:Union[Tuple[TensArr, ...], List[TensArr]]
    axis=0
    astype: type=None
    """
    return _cat_stack('cat', *args, **kargs)


def stack(*args, **kargs) -> TensArr:
    """
    <parameters>
    a: Union[Tuple[TensArr, ...], List[TensArr]]
    axis=0
    astype: type=None
    """
    return _cat_stack('stack', *args, **kargs)


def sum_axis(a: TensArr, axis=None):
    if axis:
        if type(a) == Tensor:
            return a.sum(dim=axis)
        elif type(a) == ndarray:
            return a.sum(axis=axis)
        else:
            raise TypeError
    else:
        return a.sum()
