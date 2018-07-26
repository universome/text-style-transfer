import torch.nn as nn


class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def forward(self, logits_real, logits_fake):
        return logits_real.mean() - logits_fake.mean()
