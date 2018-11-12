import numpy as np

import torch

import os
import gc


def static_vars(**kwargs):
    def decorate(func):
        for k, a in kwargs.items():
            setattr(func, k, a)
        return func
    return decorate


class MultipleOptimizer(object):
    def __init__(self, *op):
        self._optimizers = [item for item in op if item]

    def zero_grad(self):
        for op in self._optimizers:
            op.zero_grad()

    def step(self):
        for op in self._optimizers:
            op.step()

    def __len__(self):
        return len(self._optimizers)

    def __getitem__(self, idx: int):
        return self._optimizers[idx]


class MultipleScheduler(object):
    def __init__(self, cls_scheduler: type,
                 optimizers: MultipleOptimizer, **kwargs):
        self._schedulers = []
        for op in optimizers:
            self._schedulers.append(cls_scheduler(op, **kwargs))

    def step(self):
        for sch in self._schedulers:
            sch.step()

    def __len__(self):
        return len(self._schedulers)

    def __getitem__(self, idx: int):
        return self._schedulers[idx]


def arr2str(a):
    return np.arr2str(a, formatter={'float_kind': lambda x: f'{x:.2e}'})


def print_progress(iteration: int, total: int, prefix='', suffix='',
                   decimals=1, barLength=0):
    """
    Print Progress Bar
    """
    percent = f'{100 * iteration / total:>{decimals+4}.{decimals}f}'
    if barLength == 0:
        barLength = min(os.get_terminal_size().columns, 80) \
            - len(prefix) - len(percent) - len(suffix) - 11

    filledLength = barLength * iteration // total
    bar = '#' * filledLength + '-' * (barLength - filledLength)

    print(f'{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print('')


def print_cuda_tensors():
    """
    Print All CUDA Tensors
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) \
                    or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        finally:
            pass
