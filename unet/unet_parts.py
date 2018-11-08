# sub-parts of the U-Net model

import pdb  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple


def force_size_same(a: torch.Tensor,
                    b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    diffX = a.shape[-2] - b.shape[-2]
    diffY = a.shape[-1] - b.shape[-1]

    if diffY > 0:
        b = F.pad(b, (diffY//2, int(np.ceil(diffY/2)), 0, 0))
    elif diffY < 0:
        a = F.pad(a, ((-diffY)//2, int(np.ceil((-diffY)/2)), 0, 0))

    if diffX > 0:
        b = F.pad(b, (0, 0, diffX//2, int(np.ceil(diffX/2))))
    elif diffX < 0:
        a = F.pad(a, (0, 0, (-diffX)//2, int(np.ceil((-diffX)/2))))

    return (a, b)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (5, 3), padding=(2, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (3, 2), stride=(3, 2)),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2,
                                         (3, 2), stride=(3, 2))

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x_decode, x_skip):
        x_decode = self.up(x_decode)
        x_decode, x_skip = force_size_same(x_decode, x_skip)

        x = torch.cat([x_skip, x_decode], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 11, 1)
        self.conv2 = nn.Conv2d(out_ch + 11, 11, 1)
        # self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x_decode, x_skip):
        x_decode = self.conv1(x_decode)
        # diffX = x_decode.shape[-2] - x_skip.shape[-2]
        # diffY = x_decode.shape[-1] - x_skip.shape[-1]
        # if diffY > 0:
        #     x_skip = F.pad(x_skip,
        #                    (diffY//2, int(np.ceil(diffY/2)), 0, 0))
        # elif diffY < 0:
        #     x_decode = F.pad(x_decode,
        #                      ((-diffY)//2, int(np.ceil((-diffY)/2)), 0, 0))
        #
        # if diffX > 0:
        #     x_skip = F.pad(x_skip,
        #                    (0, 0, diffX//2, int(np.ceil(diffX/2))))
        # elif diffX < 0:
        #     x_decode = F.pad(x_decode,
        #                      (0, 0, (-diffX)//2, int(np.ceil((-diffX)/2))))

        x = torch.cat([x_skip, x_decode], dim=1)
        mask = self.conv2(x)
        dim_ch = 1 if x_skip.dim() == 4 else 0

        mask[..., 0:9, :, :] = self.tanh(mask[..., 0:9, :, :])
        mask[..., 9:, :, :] = self.relu(mask[..., 9:, :, :])
        y = [None]*4
        for idx in range(0, 9, 3):
            y[idx//3] = (mask[..., idx:idx+3, :, :]
                         * x_skip[..., 0:3, :, :]).sum(dim=dim_ch,
                                                       keepdim=True)

        y[3] = mask[..., 9:10, :, :] * x_skip[..., 3:4, :, :]
        y = torch.cat(y, dim=dim_ch)
        y[..., 0:3, :, :] *= mask[..., 10:11, :, :]  # Broadcast
        return y
