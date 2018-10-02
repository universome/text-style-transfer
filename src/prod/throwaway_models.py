from typing import List

import torch
import torch.nn as nn
from firelab.utils.training_utils import cudable


class WeightedEnsemble(nn.Module):
    "Simple weighted ensemble"
    def __init__(self, models:List[nn.Module], weights:List[float]):
        assert len(models) == len(weights)

        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, *args, **kwargs):
        predictions = [m(*args, **kwargs) for m in self.models]
        predictions = [w * p for w,p in zip(self.weights, predictions)]
        predictions = sum(predictions)

        return predictions


class CharLMFromEmbs(nn.Module):
    "RNNLM with style embeds"
    def __init__(self, rnn_lm, style_embed, n_layers):
        super(CharLMFromEmbs, self).__init__()

        self.n_layers = n_layers
        self.lm = rnn_lm
        self.style_embed = style_embed

    def forward(self, z, x, return_z=False):
        return self.lm(z, x, return_z=return_z)

    def init_z(self, batch_size, style):
        styles = cudable(torch.ones(self.n_layers, batch_size).fill_(style)).long()
        z = self.style_embed(styles)

        return z
