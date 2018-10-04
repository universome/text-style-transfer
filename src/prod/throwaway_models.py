from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from firelab.utils.training_utils import cudable


class WeightedLMEnsemble(nn.Module):
    "Simple weighted ensemble"
    def __init__(self, models:List[nn.Module], weights:List[float]):
        super(WeightedLMEnsemble, self).__init__()

        assert len(models) == len(weights)
        assert sum(weights) == 1

        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, zs, x, return_z=False):
        if return_z:
            results = [m(z, x, return_z) for m, z in zip(self.models, zs)]
            logits = [r[0] for r in results]
            zs = torch.stack([r[1] for r in results])

            return self.weight(logits), zs
        else:
            logits = [m(z, x) for m, z in zip(self.models, zs)]

            return self.weight(logits)

    def weight(self, logits):
        probs = [F.softmax(l, dim=2) for l in logits]
        predictions = sum([w * p for w,p in zip(self.weights, probs)])

        return predictions.log()

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
