# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
from .unet_parts import InConv, DownAndConv, UpAndConv, OutConvMap


class UNet(nn.Module):
    def __init__(self, ch_in, ch_out, ch_base=64):
        super(UNet, self).__init__()
        self.inc = InConv(ch_in, ch_base)

        self.downs = nn.ModuleList(
            [DownAndConv(ch_base*(2**ii), ch_base*(2**(ii + 1))) for ii in range(0, 4, 1)]
        )
        self.ups = nn.ModuleList(
            [UpAndConv(ch_base*(2**ii), ch_base*(2**(ii - 1))) for ii in range(4, 0, -1)]
        )

        # self.outc = OutConv(ch_base, ch_out)
        self.outc = OutConvMap(ch_base, ch_out)

    def forward(self, xin):
        x_skip = [torch.Tensor]*4
        x_skip[0] = self.inc(xin)

        for idx, down in enumerate(self.downs[:-1]):
            x_skip[idx+1] = down(x_skip[idx])

        x = self.downs[-1](x_skip[-1])

        for idx, up in zip(range(3, -1, -1), self.ups):
            x = up(x, x_skip[idx])

        x = self.outc(x)
        return x
