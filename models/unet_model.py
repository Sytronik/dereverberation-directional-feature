"""
Based on FusionNet
"""

import torch
from torch import nn
from .unet_parts import InConv, DownAndConv, UpAndConv, OutConv


class UNet(nn.Module):
    def __init__(self, ch_in, ch_out, ch_base=32, depth=4, kernel_size=(3, 3)):
        super().__init__()
        self.inc = InConv(ch_in, ch_base,
                          kernel_size=kernel_size)

        self.downs = nn.ModuleList(
            [DownAndConv(ch_base * (2**ii), ch_base * (2**(ii + 1)),
                         kernel_size=kernel_size)
             for ii in range(0, depth)]
        )
        self.ups = nn.ModuleList(
            [UpAndConv(ch_base * (2**(ii + 1)), ch_base * (2**ii),
                       kernel_size=kernel_size)
             for ii in reversed(range(0, depth))]
        )

        self.outc = OutConv(ch_base, ch_out)

    def forward(self, xin):
        xs_skip = [self.inc(xin)]

        for down in self.downs[:-1]:
            xs_skip.append(down(xs_skip[-1]))

        x = self.downs[-1](xs_skip[-1])

        for item_skip, up in zip(reversed(xs_skip), self.ups):
            x = up(x, item_skip)

        x = self.outc(x)
        return x
