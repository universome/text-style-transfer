import torch
import torch.nn as nn

from .encoder import Encoder


class Critic(nn.Module):
    def __init__(self, config, vocab):
        super(Critic, self).__init__()

        self.encoder = Encoder(config, vocab)
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, x, mask=None, onehot=True):
        embs, _ = self.encoder(x, mask=mask, onehot=onehot)
        z = embs.sum(dim=1)
        logits = self.classifier(z)

        return logits
