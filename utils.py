import numpy as np

import torch

import os
import gc


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = [item for item in op if item]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def __len__(self):
        return len(self.optimizers)

    def __getitem__(self, idx: int):
        return self.optimizers[idx]


class MultipleScheduler(object):
    def __init__(self, cls_scheduler: type,
                 optimizers: MultipleOptimizer, **kwargs):
        self.schedulers = []
        for op in optimizers:
            self.schedulers.append(cls_scheduler(op, **kwargs))

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def __len__(self):
        return len(self.schedulers)

    def __getitem__(self, idx: int):
        return self.schedulers[idx]


def arr2str(a):
    return np.arr2str(a, formatter={'float_kind': lambda x: f'{x:.2e}'})


def printProgress(iteration: int, total: int, prefix='', suffix='',
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
