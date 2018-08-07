import torch
import torch.nn as nn


class WCriticLoss(nn.Module):
    def __init__(self):
        super(WCriticLoss, self).__init__()

    def forward(self, logits_real, logits_fake):
        return logits_fake.mean() - logits_real.mean()


class WGeneratorLoss(nn.Module):
    def __init__(self):
        super(WGeneratorLoss, self).__init__()

    def forward(self, logits_fake):
        return -logits_fake.mean()


class DiscriminatorLoss(nn.Module):
    """
    Traditional discriminator loss
    """
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits_real, logits_fake):
        loss_on_real = self.criterion(logits_real, torch.zeros_like(logits_real))
        loss_on_fake = self.criterion(logits_fake, torch.ones_like(logits_fake))

        return (loss_on_real + loss_on_fake) / 2
