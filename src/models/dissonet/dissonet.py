import torch.nn as nn


class DissoNet(nn.Module):
    """Wrapper for the architecture for better data parallelism"""
    def __init__(self, encoder, decoder, critic, merge_nn):
        super(DissoNet, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.merge_nn = merge_nn

    def forward(self, domain_x, domain_y):
        state_domain_x = self.encoder(domain_x)
        state_domain_y = self.encoder(domain_y)

        hid_domain_x = self.merge_nn(state_domain_x, 0)
        hid_domain_y = self.merge_nn(state_domain_y, 1)

        recs_x = self.decoder(hid_domain_x, domain_x[:, :-1])
        recs_y = self.decoder(hid_domain_y, domain_y[:, :-1])

        critic_preds_x = self.critic(state_domain_x)
        critic_preds_y = self.critic(state_domain_y)

        return recs_x, recs_y, critic_preds_x, critic_preds_y
