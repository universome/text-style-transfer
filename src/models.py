import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, input_size, n_hid_layers, hid_size=1024):
        super(FFN, self).__init__()

        self.n_hid_layers = n_hid_layers
        self.hid_size = hid_size
        self.input_size = input_size

        self.nn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.input_size, self.hid_size),
            nn.ReLU(),
            self._init_hidden(),
            nn.Linear(self.hid_size, 1),
            nn.Sigmoid()
        )

    def _init_hidden(self):
        layers = []

        for _ in range(self.n_hid_layers):
            layers += [
                nn.Dropout(0.3),
                nn.Linear(self.hid_size, self.hid_size),
                nn.ReLU()
            ]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
