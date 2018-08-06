import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils import cudable
from sklearn.model_selection import train_test_split

from src.models.dissonet import RNNEncoder, RNNDecoder, MergeNN
from src.models import FFN
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.gan_losses import WCriticLoss, DiscriminatorLoss
from src.inference import inference


class DissoNetTrainer(BaseTrainer):
    def __init__(self, config):
        super(DissoNetTrainer, self).__init__(config)

        self.losses['val_rec_loss'] = [] # Here we'll write history for early stopping

    def init_dataloaders(self):
        batch_size = self.config.batch_size
        project_path = self.config.firelab.project_path
        domain_x_data_path = os.path.join(project_path, self.config.data.domain_x)
        domain_y_data_path = os.path.join(project_path, self.config.data.domain_y)

        with open(domain_x_data_path) as f: domain_x = f.read().splitlines()
        with open(domain_y_data_path) as f: domain_y = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        fields = [('domain_x', text), ('domain_y', text)]
        examples = [Example.fromlist([m,o], fields) for m,o in zip(domain_x, domain_y)]
        train_exs, val_exs = train_test_split(examples, test_size=self.config.val_set_size,
                                              random_state=self.config.random_seed)

        self.train_ds, self.val_ds = Dataset(train_exs, fields), Dataset(val_exs, fields)
        text.build_vocab(self.train_ds, max_size=self.config.hp.get('max_vocab_size'))

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, repeat=False, shuffle=False)

    def init_models(self):
        emb_size = self.config.hp.emb_size
        hid_size = self.config.hp.hid_size
        voc_size = len(self.vocab)
        dropout_p = self.config.hp.dropout
        dropword_p = self.config.hp.dropword

        self.encoder = cudable(RNNEncoder(emb_size, hid_size, voc_size, dropword_p))
        self.decoder = cudable(RNNDecoder(emb_size, hid_size, voc_size, dropword_p))
        self.critic = cudable(FFN([hid_size, 1], dropout=dropout_p))
        self.merge_nn = cudable(MergeNN(hid_size))

    def init_criterions(self):
        self.rec_criterion = cross_entropy_without_pads(self.vocab)
        self.critic_criterion = WCriticLoss()

    def init_optimizers(self):
        self.critic_optim = Adam(self.critic.parameters(), lr=self.config.hp.lr.critic)
        ae_params = chain(self.encoder.parameters(), self.decoder.parameters(), self.merge_nn.parameters())
        self.ae_optim = Adam(ae_params, lr=self.config.hp.lr.ae)

    def train_on_batch(self, batch):
        rec_loss, critic_loss, ae_loss = self.loss_on_batch(batch)

        self.ae_optim.zero_grad()
        ae_loss.backward(retain_graph=True)
        self.ae_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.writer.add_scalar('Rec loss', rec_loss, self.num_iters_done)
        self.writer.add_scalar('Critic loss', critic_loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        # Computing codes we need
        state_domain_x = self.encoder(batch.domain_x)
        state_domain_y = self.encoder(batch.domain_y)

        hid_domain_x = self.merge_nn(state_domain_x, 1)
        hid_domain_y = self.merge_nn(state_domain_y, 1)

        # Reconstructing
        recs_domain_x = self.decoder(hid_domain_x, batch.domain_x[:, :-1])
        recs_domain_y = self.decoder(hid_domain_y, batch.domain_y[:, :-1])

        # Computing reconstruction loss
        rec_loss_domain_x = self.rec_criterion(recs_domain_x.view(-1, len(self.vocab)), batch.domain_x[:, 1:].contiguous().view(-1))
        rec_loss_domain_y = self.rec_criterion(recs_domain_y.view(-1, len(self.vocab)), batch.domain_y[:, 1:].contiguous().view(-1))
        rec_loss = (rec_loss_domain_x + rec_loss_domain_y) / 2

        # Computing critic loss
        critic_domain_x_preds, critic_domain_y_preds = self.critic(state_domain_x), self.critic(state_domain_y)
        critic_loss = self.critic_criterion(critic_domain_x_preds, critic_domain_y_preds)

        # Loss for encoder and decoder is threefold
        coefs = self.config.loss_coefs
        ae_loss = coefs.rec * rec_loss - coefs.critic * critic_loss

        return rec_loss, critic_loss, ae_loss

    def validate(self):
        rec_losses = []
        critic_losses = []

        for batch in self.val_dataloader:
            rec_loss, critic_loss, _ = self.loss_on_batch(batch)
            rec_losses.append(rec_loss.item())
            critic_losses.append(critic_loss.item())

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.writer.add_scalar('val_critic_loss', np.mean(critic_losses), self.num_iters_done)

        self.losses['val_rec_loss'].append(np.mean(rec_losses))

        # Ok, let's now validate style transfer and auto-encoding
        self.validate_inference()

    def validate_inference(self):
        """
        Performs inference on a val dataloader
        (computes predictions without teacher's forcing)
        """
        x2y, y2x, x2x, y2y, gx, gy = self.transfer_style(self.val_dataloader)

        x2y_bleu = compute_bleu_for_sents(x2y, gx)
        y2x_bleu = compute_bleu_for_sents(y2x, gy)
        x2x_bleu = compute_bleu_for_sents(x2x, gx)
        y2y_bleu = compute_bleu_for_sents(y2y, gy)

        self.writer.add_scalar('x2y val BLEU', x2y_bleu, self.num_iters_done)
        self.writer.add_scalar('y2x val BLEU', y2x_bleu, self.num_iters_done)
        self.writer.add_scalar('x2x val BLEU', x2x_bleu, self.num_iters_done)
        self.writer.add_scalar('y2y val BLEU', y2y_bleu, self.num_iters_done)

        # Ok, let's log generated sequences
        texts = [get_text_from_sents(*sents) for sents in zip(x2y, y2x, x2x, y2y, gx, gy)]
        texts = texts[:10] # If we'll write too many texts, nothing will be displayed in TB
        text = '\n===================\n'.join(texts)

        self.writer.add_text('Generated examples', text, self.num_iters_done)

    def transfer_style(self, dataloader):
        """
        Produces predictions for a given dataloader
        """
        domain_x_to_domain_y = []
        domain_y_to_domain_x = []
        domain_x_to_domain_x = []
        domain_y_to_domain_y = []
        gold_domain_x = []
        gold_domain_y = []

        for batch in dataloader:
            x2y, y2x, x2x, y2y = self.transfer_style_on_batch(batch)

            domain_x_to_domain_y.extend(x2y)
            domain_y_to_domain_x.extend(y2x)
            domain_x_to_domain_x.extend(x2x)
            domain_y_to_domain_y.extend(y2y)

            gold_domain_x.extend(batch.domain_x.detach().cpu().numpy().tolist())
            gold_domain_y.extend(batch.domain_y.detach().cpu().numpy().tolist())

        # Converting to sentences
        x2y_sents = itos_many(domain_x_to_domain_y, self.vocab)
        y2x_sents = itos_many(domain_y_to_domain_x, self.vocab)
        x2x_sents = itos_many(domain_x_to_domain_x, self.vocab)
        y2y_sents = itos_many(domain_y_to_domain_y, self.vocab)
        gx_sents = itos_many(gold_domain_x, self.vocab)
        gy_sents = itos_many(gold_domain_y, self.vocab)

        return x2y_sents, y2x_sents, x2x_sents, y2y_sents, gx_sents, gy_sents

    def transfer_style_on_batch(self, batch):
        state_domain_x = self.encoder(batch.domain_x)
        state_domain_y = self.encoder(batch.domain_y)

        domain_x_to_domain_y_z = self.merge_nn(state_domain_x, 0)
        domain_y_to_domain_x_z = self.merge_nn(state_domain_y, 1)
        domain_x_to_domain_x_z = self.merge_nn(state_domain_x, 1)
        domain_y_to_domain_y_z = self.merge_nn(state_domain_y, 0)

        x2y = inference(self.decoder, domain_x_to_domain_y_z, self.vocab)
        y2x = inference(self.decoder, domain_y_to_domain_x_z, self.vocab)
        x2x = inference(self.decoder, domain_x_to_domain_x_z, self.vocab)
        y2y = inference(self.decoder, domain_y_to_domain_y_z, self.vocab)

        return x2y, y2x, x2x, y2y


def get_text_from_sents(x2y_s, y2x_s, x2x_s, y2y_s, gx_s, gy_s):
    # TODO: Move this somewhere from trainer file? Or create some nice template?
    return """
        Gold X: [{}]
        Gold Y: [{}]

        x2y: [{}]
        y2x: [{}]

        x2x: [{}]
        y2y: [{}]
    """.format(gx_s, gy_s, x2y_s, y2x_s, x2x_s, y2y_s)
