import torch
import torch.nn as nn


class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def forward(self, logits_real, logits_fake):
        return logits_real.mean() - logits_fake.mean()


class DiscriminatorLoss(nn.Module):
    """
    Traditional discriminator loss
    """
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits_real, logits_fake):
        targets_real = torch.zeros_like(logits_real)
        targets_fake = torch.ones_like(logits_real)

        loss_on_real = self.criterion(logits_real, targets_real)
        loss_on_fake = self.criterion(logits_fake, targets_fake)

        return (loss_on_real + loss_on_fake) / 2
