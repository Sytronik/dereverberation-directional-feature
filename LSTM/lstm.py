import numpy as np

import torch
from torch import nn


class ContextLSTM(nn.Module):
    def __init__(self, L_feature, T_input, T_output, bias=True):
        super().__init__()
        self.T_context = T_output - T_input
        self.T_input = T_input
        self.input_size = T_input*L_feature
        self.output_size = T_output*L_feature
        self.lstm = nn.LSTMCell(self.input_size, self.output_size, bias=bias)

    def forward(self, pad_seq, T_samples=None, n_batches=None):
        # pad_seq: B, max_T, (CxF)
        B, max_T = pad_seq.shape[0:1]
        max_T_out = max_T - self.T_context
        y = torch.zeros_like(pad_seq[:, :max_T_out])

        if T_samples:
            T_samples = np.append(np.asarray(T_samples), 0)  # ensure type
            T_diff = np.flip(np.diff(-T_samples))
            T_diff[0] -= self.T_context
            n_batches = np.repeat(np.arange(B, 0, -1), T_diff)
        elif n_batches:
            n_batches = np.asarray(n_batches)[self.T_context:]  # ensure type
        else:
            raise TypeError

        c = torch.zeros(self.input_size)
        h = torch.zeros(self.input_size)
        for i, b in enumerate(n_batches):
            y[:b, i], c = self.lstm(pad_seq[:b, i:i+self.T_input].view(b, -1), (h, c))
            h = y[:b, i]

        return y, n_batches


class MyLSTM(nn.Module):
    def __init__(self, L_feature, T_input, T_hidden, T_output):
        super(MyLSTM, self).__init__()
        self.lstm1 = ContextLSTM(L_feature*T_input, L_feature*T_hidden)
        self.lstm2 = ContextLSTM(L_feature*T_hidden, L_feature*T_output)
        self.T_context = T_input - T_output

    def forward(self, pad_seq, T_samples):
        x, n_batches = self.lstm1(pad_seq, T_samples=T_samples)
        x, _ = self.lstm2(x, n_batches=n_batches)

        return x
