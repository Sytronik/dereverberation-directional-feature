"""
sub-parts of the FusionNet model
"""

from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def force_size_same(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    diffX = a.shape[-2] - b.shape[-2]
    diffY = a.shape[-1] - b.shape[-1]

    if diffY > 0:
        b = F.pad(b, [diffY // 2, int(np.ceil(diffY / 2)), 0, 0])
    elif diffY < 0:
        a = F.pad(a, [(-diffY) // 2, int(np.ceil((-diffY) / 2)), 0, 0])

    if diffX > 0:
        b = F.pad(b, [0, 0, diffX // 2, int(np.ceil(diffX / 2))])
    elif diffX < 0:
        a = F.pad(a, [0, 0, (-diffX) // 2, int(np.ceil((-diffX) / 2))])

    return a, b


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, act_fn, *,
                 kernel_size=(3, 3), padding=None, groups=1, stride=(1, 1)):
        super().__init__()
        if not padding:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.cba = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding, groups=groups, stride=stride),
            nn.BatchNorm2d(out_ch),
        )
        if act_fn:
            self.cba.add_module('2', act_fn)

    def forward(self, x):
        return self.cba(x)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_fn: nn.Module, *,
                 kernel_size=(3, 3), padding=None, last_act_fn=True):
        super().__init__()
        self.skipcba = ConvBNAct(in_ch, out_ch, None, kernel_size=(1, 1))

        self.cba1 = ConvBNAct(in_ch, out_ch, act_fn,
                              kernel_size=kernel_size, padding=padding,
                              )
        self.cba2 = ConvBNAct(out_ch, out_ch, None,
                              kernel_size=kernel_size, padding=padding,
                              )

        self.act_fn = act_fn if last_act_fn else None

    def forward(self, x):
        residual = self.skipcba(x)

        out = self.cba1(x)
        out = self.cba2(out)
        out += residual
        if self.act_fn:
            out = self.act_fn(out)

        return out


class FusionNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_fn: nn.Module,
                 kernel_size=(3, 3), padding=None,
                 ):
        super().__init__()
        self.cba1 = ConvBNAct(in_ch, out_ch, act_fn,
                              kernel_size=kernel_size, padding=padding,
                              )
        self.resblock = ResidualBlock(out_ch, out_ch, act_fn, last_act_fn=False,
                                      kernel_size=kernel_size, padding=padding,
                                      )
        self.cba2 = ConvBNAct(out_ch, out_ch, act_fn,
                              kernel_size=kernel_size, padding=padding,
                              )

    def forward(self, x):
        x = self.cba1(x)
        x = self.resblock(x)
        x = self.cba2(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *,
                 kernel_size=(3, 3), padding=None,
                 ):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, nn.ReLU(inplace=True),
                                   kernel_size=kernel_size, padding=padding,
                                   )

    def forward(self, x):
        x = self.block(x)
        return x


class DownAndConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *,
                 kernel_size=(3, 3), padding=None,
                 ):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 2))

        self.block = FusionNetBlock(in_ch, out_ch, nn.ReLU(inplace=True),
                                    kernel_size=kernel_size, padding=padding,
                                    )

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class UpAndConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *,
                 kernel_size=(3, 3), padding=None,
                 ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, (2, 2), stride=(2, 2))
        self.block = FusionNetBlock(out_ch, out_ch, nn.ReLU(inplace=True),
                                    kernel_size=kernel_size, padding=padding,
                                    )

    def forward(self, x, x_skip):
        x = self.up(x)
        x, x_skip = force_size_same(x, x_skip)

        out = (x + x_skip) / 2
        out = self.block(out)
        return out


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1))
        # self.act_fn = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        # x = 2 * self.act_fn(x)
        return x
