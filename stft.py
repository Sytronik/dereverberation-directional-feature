import numpy as np
import scipy.signal as scsig
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    __slots__ = ('filter_length', 'hop_length', 'forward_basis', 'inverse_basis')

    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        scale = self.filter_length / self.hop_length
        window = scsig.hann(self.filter_length, sym=False)[:, np.newaxis].T
        # window = window/window.sum()
        fourier_basis = np.fft.fft(np.eye(self.filter_length)) * window

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.tensor(fourier_basis[:, None, :],
                                     dtype=torch.float32, requires_grad=False)
        inverse_basis = torch.tensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :],
                                     dtype=torch.float32, requires_grad=False)

        self.register_buffer('forward_basis', forward_basis)
        self.register_buffer('inverse_basis', inverse_basis)

    # B, n -> B, F, T
    def transform(self, wav):
        num_batches = wav.size(0)
        num_samples = wav.size(1)

        wav = wav.view(num_batches, 1, num_samples)
        spec_realimag = F.conv1d(wav,
                                 self.forward_basis,
                                 stride=self.hop_length,
                                 padding=self.filter_length,
                                 )
        cutoff = int((self.filter_length / 2) + 1)
        real = spec_realimag[:, :cutoff, 1:-1]
        imag = spec_realimag[:, cutoff:, 1:-1]

        mag = (real**2 + imag**2)**0.5
        phase = torch.atan2(imag, real)
        return mag, phase

    # B, F, T -> B, n'
    def inverse(self, mag, phase):
        spec_realimag = torch.cat([mag * torch.cos(phase),
                                   mag * torch.sin(phase)], dim=1)

        wav = F.conv_transpose1d(spec_realimag,
                                 self.inverse_basis,
                                 stride=self.hop_length,
                                 padding=0,
                                 )
        wav = wav[:, 0, self.hop_length:-self.hop_length]
        # wav = wav[:, :, :self.num_samples]
        return wav

    def forward(self, wav):
        mag, phase = self.transform(wav)
        recon = self.inverse(mag, phase)
        return recon
