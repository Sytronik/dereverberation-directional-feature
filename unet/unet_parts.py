# sub-parts of the U-Net model

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 2), stride=(1, 2)),
            nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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
            nn.Conv2d(in_ch, in_ch, (2, 2), stride=(2, 2)),
            # nn.MaxPool2d((2, 2)),
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
                                         (2, 2), stride=(2, 2))

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
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch + 11, 11, 1)

        self.act_transform = nn.Softsign()
        # self.act_transform = nn.Tanh()

        # self.act_mask = nn.ReLU()
        # self.act_mask = nn.PReLU(num_parameters=1, init=0.25)
        # self.act_mask = nn.Softplus(beta=1, threshold=20)
        self.act_mask = nn.Sigmoid()

    def forward(self, x_decode, x_skip):
        x_decode = self.conv1(x_decode)
        x_decode = self.relu1(x_decode)
        x = torch.cat([x_skip, x_decode], dim=1)
        out = self.conv2(x)
        einexp = 'bcft,bcft->bft' if x_skip.dim() == 4 else 'cft,cft->ft'

        transform = self.act_transform(out[..., :9, :, :])
        mask = self.act_mask(out[..., 9:, :, :])
        y = torch.zeros_like(x_skip)
        for idx in range(3):
            y[..., idx, :, :] = torch.einsum(
                einexp,
                (transform[..., 3*idx:3*idx+3, :, :],
                 x_skip[..., :3, :, :])
            )
        y[..., :3, :, :] *= mask[..., :1, :, :]  # Broadcast
        y[..., 3, :, :] = mask[..., 1, :, :] * x_skip[..., 3, :, :]
        return y
