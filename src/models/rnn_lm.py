import torch
import torch.nn as nn


class RNNLM(nn.Module):
    def __init__(self, size, vocab):
        super(RNNLM, self).__init__()

        self.embed = nn.Embedding(len(vocab), size, padding_idx=vocab.stoi['<pad>'])
        self.gru = nn.GRU(size, size, batch_first=True)
        self.out = nn.Linear(size, len(vocab))

    def forward(self, z, x):
        x = self.embed(x)
        states, _ = self.gru(x, z.unsqueeze(0))
        out = self.out(states)

        return out
