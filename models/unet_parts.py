# sub-parts of the U-Net model

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def force_size_same(
        a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    diffX: int = a.shape[-2] - b.shape[-2]
    diffY: int = a.shape[-1] - b.shape[-1]

    if diffY > 0:
        b = F.pad(b, (diffY//2, int(np.ceil(diffY/2)), 0, 0))
    elif diffY < 0:
        a = F.pad(a, ((-diffY)//2, int(np.ceil((-diffY)/2)), 0, 0))

    if diffX > 0:
        b = F.pad(b, (0, 0, diffX//2, int(np.ceil(diffX/2))))
    elif diffX < 0:
        a = F.pad(a, (0, 0, (-diffX)//2, int(np.ceil((-diffX)/2))))

    return a, b


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_fn: nn.Module, last_act_fn=True):
        super(ResidualBlock, self).__init__()
        self.cba1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            act_fn
        )
        self.cba2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch)
        )
        self.down_ch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.last_act_fn = last_act_fn

    def forward(self, x):
        residual = self.down_ch(x)
        out = self.cba1(x)
        out = self.cba2(out)
        out += residual
        if self.last_act_fn:
            out = self.act_fn(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_fn: nn.Module):
        super(ConvBlock, self).__init__()
        # self.cba1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1)),
        #     nn.BatchNorm2d(out_ch),
        #     act_fn,
        # )
        self.resblock = ResidualBlock(in_ch, out_ch, act_fn, last_act_fn=False)
        # self.cba2 = nn.Sequential(
        #     # nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 2), stride=(1, 2)),
        #     nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1)),
        #     nn.BatchNorm2d(out_ch),
        #     act_fn,
        # )

    def forward(self, x):
        # x = self.cba1(x)
        x = self.resblock(x)
        # x = self.cba2(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(InConv, self).__init__()
        self.block = ConvBlock(in_ch, out_ch, nn.ReLU())
        # self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.block(x)
        return x


class DownAndConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(DownAndConv, self).__init__()
        self.pool = nn.MaxPool2d((2, 2))

        self.block = ConvBlock(in_ch, out_ch, nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class UpAndConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear=False):
        super(UpAndConv, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, (2, 2), stride=(2, 2))

        self.block = ConvBlock(out_ch, out_ch, nn.ReLU())
        # self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x_decode, x_skip):
        x_decode = self.up(x_decode)
        x_decode, x_skip = force_size_same(x_decode, x_skip)

        # x = torch.cat([x_skip, x_decode], dim=1)
        x = (x_skip + x_decode)/2
        x = self.block(x)
        return x


# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#         self.conv1 = nn.Conv2d(in_ch, 11, 1)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_ch + 11, 11, 1)
#
#         self.act_transform = nn.Softsign()
#         # self.act_transform = nn.Tanh()
#
#         # self.act_mask = nn.ReLU()
#         # self.act_mask = nn.PReLU(num_parameters=1, init=0.25)
#         # self.act_mask = nn.Softplus(beta=1, threshold=20)
#         self.act_mask = nn.Sigmoid()
#
#     def forward(self, x_decode, x_skip):
#         x_decode = self.conv1(x_decode)
#         x_decode = self.relu1(x_decode)
#         x = torch.cat([x_skip, x_decode], dim=1)
#         out = self.conv2(x)
#         einexp = 'bcft,bcft->bft' if x_skip.dim() == 4 else 'cft,cft->ft'
#
#         transform = self.act_transform(out[..., :9, :, :])
#         mask = self.act_mask(out[..., 9:, :, :])
#         y = torch.empty_like(x_skip)
#         for idx in range(3):
#             y[..., idx, :, :] = torch.einsum(
#                 einexp,
#                 [transform[..., 3*idx:3*idx + 3, :, :], x_skip[..., :3, :, :]]
#             )
#         y[..., :3, :, :] *= mask[..., :1, :, :]  # Broadcast
#         y[..., 3, :, :] = mask[..., 1, :, :] * x_skip[..., 3, :, :]
#         return y


class OutConvMap(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(OutConvMap, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1))
        self.act_alpha = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act_alpha(x)
        return x
