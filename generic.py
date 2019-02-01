"""
Generic type & functions for torch.Tensor and np.ndarray
"""
from typing import Sequence, TypeVar, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

TensArr = TypeVar('TensArr', Tensor, ndarray)
TensArrOrSeq = Union[TensArr, Sequence[TensArr]]

dict_package = {Tensor: torch, ndarray: np}
dict_cat_stack_fn = {(Tensor, 'cat'): torch.cat,
                     (ndarray, 'cat'): np.concatenate,
                     (Tensor, 'stack'): torch.stack,
                     (ndarray, 'stack'): np.stack,
                     }


def convert_dtype(dtype: type, pkg) -> type:
    if hasattr(dtype, '__name__'):
        if pkg == np:
            return dtype
        else:
            return eval(f'torch.{dtype.__name__}')
    else:
        if pkg == np:
            return eval(f'np.{str(dtype).split(".")[-1]}')
        else:
            return dtype


def copy(a: TensArr, requires_grad=True) -> TensArr:
    if type(a) == Tensor:
        return a.clone() if requires_grad else torch.tensor(a)
    elif type(a) == ndarray:
        return np.copy(a)
    else:
        raise TypeError


def arctan2(a: TensArr, b: TensArr, out: TensArr = None) -> TensArr:
    if type(a) == Tensor:
        return torch.atan2(a, b, out=out)
    else:
        return np.arctan2(a, b, out=out)


def convert(a: TensArr, astype: type, device: Union[int, torch.device] = None) -> TensArr:
    if astype == Tensor:
        if type(a) == Tensor:
            return a.to(device)
        else:
            return torch.as_tensor(a, dtype=torch.float32, device=device)
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


def transpose(a: TensArr, axes: Union[int, Sequence[int]] = None) -> TensArr:
    if type(a) == Tensor:
        if not axes:
            if a.dim() >= 2:
                return a.permute((1, 0) + tuple(range(2, a.dim())))
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


def einsum(subscripts: str,
           operands: Sequence[TensArr],
           astype: type = None) -> TensArr:
    if not astype:
        astype = type(operands[0])
        if astype != Tensor and astype != ndarray:
            raise TypeError
    else:
        types = [type(item) for item in operands]
        for idx, type_ in enumerate(types):
            if type_ != astype:
                if type(operands) != list:
                    operands = list(operands)
                operands[idx] = convert(operands[idx], astype)

    return dict_package[astype].einsum(subscripts, operands)


def _cat_stack(fn: str,
               a: Sequence[TensArr],
               axis=0,
               astype: type = None) -> TensArr:
    types = [type(item) for item in a]
    if not astype:
        astype = types[0]
    for idx, type_ in enumerate(types):
        if type_ != astype:
            a[idx] = convert(a[idx], astype)

    return dict_cat_stack_fn[(astype, fn)](a, axis)


def cat(*args, **kargs) -> TensArr:
    """
    <parameters>
    a: Iterable[TensArr]
    axis=0
    astype: type=None
    """
    return _cat_stack('cat', *args, **kargs)


def stack(*args, **kargs) -> TensArr:
    """
    <parameters>
    a: Iterable[TensArr]
    axis=0
    astype: type=None
    """
    return _cat_stack('stack', *args, **kargs)
