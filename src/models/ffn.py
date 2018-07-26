import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_size, n_hid_layers, dropout=0, hid_size=1024, out_size=1):
        super(FFN, self).__init__()

        self.n_hid_layers = n_hid_layers
        self.hid_size = hid_size
        self.input_size = input_size
        self.dropout = dropout

        self.nn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.input_size, self.hid_size),
            nn.SELU(),
            self._init_hidden(),
            nn.Linear(self.hid_size, out_size),
        )

    def _init_hidden(self):
        layers = []

        for _ in range(self.n_hid_layers):
            layers += [
                nn.Dropout(self.dropout),
                nn.Linear(self.hid_size, self.hid_size),
                nn.SELU()
            ]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
