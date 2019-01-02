"""
Generic type & functions for torch.Tensor and np.ndarray
"""
from typing import Union, TypeVar, Sequence

import numpy as np
from numpy import ndarray

import torch
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
    i_first = str(dtype).find('\'')
    if i_first == -1:
        if pkg == torch:
            return dtype
        else:
            str_dtype = str(dtype).split('.')[-1]
            return eval(f'np.{str_dtype}')
    else:
        if pkg == torch:
            i_last = str(dtype).rfind('\'')
            str_dtype = str(dtype)[i_first + 1:i_last]
            str_dtype = str_dtype.split('.')[-1]
            return eval(f'torch.{str_dtype}')
        else:
            return dtype


def copy(a: TensArr, requires_grad=True) -> TensArr:
    if type(a) == Tensor:
        return a.clone() if requires_grad else torch.tensor(a)
    elif type(a) == ndarray:
        return np.copy(a)
    else:
        raise TypeError


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
            if type(a) != list:
                a = list(a)
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
