import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, sizes, dropout=0):
        super(FFN, self).__init__()

        self.dropout = dropout

        layers = [self.init_layer(sizes[i], sizes[i+1], (i+2) == len(sizes)) for i in range(len(sizes)-1)]
        layers = [m for l in layers for m in l] # Flattening each layer into modules

        self.nn = nn.Sequential(*layers)

    def init_layer(self, s_in, s_out, is_last_layer=False):
        layer = [nn.Dropout(self.dropout), nn.Linear(s_in, s_out)]

        if not is_last_layer:
            layer.extend([nn.SELU(), nn.BatchNorm1d(s_out)])

        return layer

    def forward(self, x):
        return self.nn(x)
