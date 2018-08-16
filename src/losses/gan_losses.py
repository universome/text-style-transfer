import torch
import torch.nn as nn
import torch.autograd as autograd
from firelab.utils import cudable


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

def wgan_gp(critic, real_data, fake_data, *critic_args, **critic_kwargs):
    "Computes gradient penalty according to WGAN-GP paper"
    assert real_data.size() == fake_data.size()

    eps = cudable(torch.rand(real_data.size(0), 1, 1))
    eps = eps.expand(real_data.size())

    interpolations = eps * real_data + (1 - eps) * fake_data
    preds = critic(interpolations, *critic_args, **critic_kwargs)

    grads = autograd.grad(
        outputs=preds,
        inputs=interpolations,
        grad_outputs=cudable(torch.ones(preds.size())),
        retain_graph=True, create_graph=True, only_inputs=True
    )[0]

    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    return gp
