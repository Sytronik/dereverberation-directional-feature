from torch import nn


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, n_input: int, n_hidden: int, n_output: int, p: float):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_input, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.layer2 = nn.Sequential(
            # nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.layer3 = nn.Sequential(
            # nn.Dropout(p=p),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden, momentum=0.1),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.output = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x
