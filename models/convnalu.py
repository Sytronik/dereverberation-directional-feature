import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class ConvNACCell(nn.Module):
    """A Neural Accumulator (NAC) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """

    def __init__(self, ch_in, ch_out, kernel_size):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out

        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.padding = self.kernel_size[0]//2, self.kernel_size[1]//2

        in_dim = ch_in * kernel_size[0] * kernel_size[1]
        out_dim = ch_out

        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('W_hat', self.W_hat)
        self.register_parameter('M_hat', self.M_hat)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        n_batch = input.shape[0]
        output = torch.empty(n_batch, self.ch_out, *input.shape[-2:], device=input.device)
        input = F.pad(input,
                      [self.padding[0], self.padding[0], self.padding[1], self.padding[1]])

        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        for i_w in range(input.shape[-1] - self.kernel_size[0] + 1):
            for i_h in range(input.shape[-2] - self.kernel_size[1] + 1):
                output[..., i_h, i_w] = F.linear(
                    input.narrow(-1, i_w, self.kernel_size[0])
                    .narrow(-2, i_h, self.kernel_size[1])
                    .reshape(n_batch, -1),
                    W,
                    self.bias
                )

        return output

    def extra_repr(self):
        return (f'in_dim={self.ch_in}, out_dim={self.ch_out}, '
                f'kernel_size={self.kernel_size}')


class ConvNALUCell(nn.Module):
    """A Neural Arithmetic Logic Unit (NALU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """

    def __init__(self, ch_in, ch_out, kernel_size):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out

        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.padding = self.kernel_size[0]//2, self.kernel_size[1]//2

        in_dim = ch_in * kernel_size[0] * kernel_size[1]
        out_dim = ch_out

        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = ConvNACCell(ch_in, ch_out, kernel_size)
        self.register_parameter('G', self.G)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=5**0.5)

    def forward(self, input):
        input = input
        a = self.nac(input)
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))

        # g
        n_batch = input.shape[0]
        g = torch.empty(n_batch, self.ch_out, *input.shape[-2:], device=input.device)
        input = F.pad(input,
                      [self.padding[0], self.padding[0], self.padding[1], self.padding[1]])
        for i_w in range(input.shape[-1] - self.kernel_size[0] + 1):
            for i_h in range(input.shape[-2] - self.kernel_size[1] + 1):
                g[..., i_h, i_w] = F.linear(
                    input.narrow(-1, i_w, self.kernel_size[0])
                    .narrow(-2, i_h, self.kernel_size[1])
                    .reshape(n_batch, -1),
                    self.G,
                    self.bias
                )
        g = torch.sigmoid(g)

        add_sub = g * a
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return (f'in_dim={self.ch_in}, out_dim={self.ch_out}, '
                f'kernel_size={self.kernel_size}')
