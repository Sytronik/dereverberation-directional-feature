import contextlib
import gc
import os
from pathlib import Path
from typing import Callable, Union, TypeVar

import numpy as np
import torch
import torch.optim
from torch import Tensor
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


TensArr = TypeVar('TensArr', Tensor, ndarray)
DICT_PACKAGE = {Tensor: torch, ndarray: np}


class DataPerDevice:
    """ convert ndarray to Tensor (in cpu or cuda devices) whenever it is needed.

    """
    __slots__ = ('data',)

    def __init__(self, data_np: ndarray):
        self.data = {ndarray: data_np}

    def __getitem__(self, typeOrtup):
        if type(typeOrtup) == tuple:
            _type, device = typeOrtup
        elif typeOrtup == ndarray:
            _type = ndarray
            device = None
        else:
            raise IndexError

        if _type == ndarray:
            return self.data[ndarray]
        else:
            if typeOrtup not in self.data:
                self.data[typeOrtup] = convert(self.data[ndarray],
                                               Tensor,
                                               device=device)
            return self.data[typeOrtup]

    def get_like(self, other: TensArr):
        if type(other) == Tensor:
            return self[Tensor, other.device]
        else:
            return self[ndarray]


def convert(a: TensArr, astype: type, device: Union[int, torch.device] = None) -> TensArr:
    """ convert Tensor to ndarray or vice versa.

    """
    if astype == Tensor:
        if type(a) == Tensor:
            return a.to(device)
        else:
            return torch.as_tensor(a, device=device)
    elif astype == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(astype)


def arr2str(a: np.ndarray, format_='e', ndigits=2) -> str:
    """convert ndarray of floats to a string expression.

    :param a:
    :param format_:
    :param ndigits:
    :return:
    """
    return np.array2string(
        a,
        formatter=dict(
            float_kind=(lambda x: f'{x:.{ndigits}{format_}}' if x != 0 else '0')
        )
    )


def print_to_file(fname: Union[str, Path], fn: Callable, args=None, kwargs=None):
    """ All `print` function calls in `fn(*args, **kwargs)`
      uses a text file `fname`.

    :param fname:
    :param fn:
    :param args: args for fn
    :param kwargs: kwargs for fn
    :return:
    """
    if fname:
        fname = Path(fname).with_suffix('.txt')

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    with (fname.open('w') if fname else open(os.devnull, 'w')) as file:
        with contextlib.redirect_stdout(file):
            fn(*args, **kwargs)


def full_extent(fig: plt.Figure, ax: plt.Axes, *extras, pad=0.01):
    """ Get the full extent of an axes, including axes labels, tick labels, and
    titles.
    
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = [ax, ax.title, ax.xaxis.label, ax.yaxis.label, *extras]
    items += [*ax.get_xticklabels(), *ax.get_yticklabels()]
    items += [e.xaxis.label for e in extras if hasattr(e, 'xaxis')]
    items += [e.yaxis.label for e in extras if hasattr(e, 'yaxis')]
    items += sum(
        (e.get_xticklabels() for e in extras if hasattr(e, 'get_xticklabels')),
        []
    )
    items += sum(
        (e.get_yticklabels() for e in extras if hasattr(e, 'get_yticklabels')),
        []
    )
    bbox = Bbox.union(
        [item.get_window_extent() for item in items if hasattr(item, 'get_window_extent')]
    )

    bbox = bbox.expanded(1.0 + pad, 1.0 + pad)
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    return bbox
