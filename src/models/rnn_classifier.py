import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, size, output_dim=1):
        super(RNNClassifier, self).__init__()

        self.gru = nn.GRU(size, size, batch_first=True)
        self.classifier = nn.Linear(size, output_dim)

    def forward(self, x):
        _, last_hidden_state = self.gru(x)
        state = last_hidden_state.squeeze(0)
        out = self.classifier(state)

        return out
