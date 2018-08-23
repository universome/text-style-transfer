import torch.nn as nn


class DissoNet(nn.Module):
    """Wrapper for the architecture for better data parallelism"""
    def __init__(self, encoder, decoder, split_nn, motivator, critic, merge_nn):
        super(DissoNet, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.split_nn = split_nn
        self.critic = critic
        self.motivator = motivator
        self.merge_nn = merge_nn

    def forward(self, domain_x, domain_y):
        state_x = self.encoder(domain_x)
        state_y = self.encoder(domain_y)

        content_x, style_x = self.split_nn(state_x)
        content_y, style_y = self.split_nn(state_y)

        hid_x = self.merge_nn(content_x, style_x)
        hid_y = self.merge_nn(content_y, style_y)

        recs_x = self.decoder(hid_x, domain_x[:, :-1])
        recs_y = self.decoder(hid_y, domain_y[:, :-1])

        critic_preds_x = self.critic(content_x)
        critic_preds_y = self.critic(content_y)

        motivator_preds_x = self.motivator(style_x)
        motivator_preds_y = self.motivator(style_y)

        return recs_x, recs_y, critic_preds_x, critic_preds_y, motivator_preds_x, motivator_preds_y
