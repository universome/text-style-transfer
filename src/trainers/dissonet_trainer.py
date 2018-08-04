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

from src.models.dissonet import RNNEncoder, RNNDecoder
from src.models import FFN
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.gan_losses import WGANLoss, DiscriminatorLoss
from src.inference import inference


class DissoNetTrainer(BaseTrainer):
    def __init__(self, config):
        super(DissoNetTrainer, self).__init__(config)

        self.losses['val_rec_loss'] = [] # Here we'll write history for early stopping

    def init_dataloaders(self):
        batch_size = self.config.get('batch_size', 16)
        project_path = self.config['firelab']['project_path']
        modern_data_path = os.path.join(project_path, self.config['data']['modern'])
        original_data_path = os.path.join(project_path, self.config['data']['original'])

        with open(modern_data_path) as f: modern = f.read().splitlines()
        with open(original_data_path) as f: original = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        fields = [('modern', text), ('original', text)]
        examples = [Example.fromlist([m,o], fields) for m,o in zip(modern, original)]
        train_exs, val_exs = train_test_split(examples, test_size=self.config['val_set_size'],
                                              random_state=self.config['random_seed'])

        self.train_ds, self.val_ds = Dataset(train_exs, fields), Dataset(val_exs, fields)
        text.build_vocab(self.train_ds, max_size=self.config['hp']['max_vocab_size'])

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, repeat=False, shuffle=False)

    def init_models(self):
        emb_size = self.config['hp'].get('emb_size')
        hid_size = self.config['hp'].get('hid_size')
        voc_size = len(self.vocab)
        dropout_p = self.config['hp']['dropout']

        self.encoder = cudable(RNNEncoder(emb_size, hid_size, voc_size))
        self.decoder = cudable(RNNDecoder(emb_size, hid_size, voc_size))
        self.critic = cudable(FFN([hid_size, 1], dropout=dropout_p))
        self.motivator = cudable(FFN([hid_size, 1], dropout=dropout_p))
        self.merge_nn = cudable(nn.Sequential(
            FFN([hid_size * 2, hid_size], dropout=dropout_p),
            nn.BatchNorm1d(hid_size)
        ))

    def init_criterions(self):
        self.rec_criterion = cross_entropy_without_pads(self.vocab)
        self.critic_criterion = WGANLoss()
        self.motivator_criterion = nn.BCEWithLogitsLoss()

    def init_optimizers(self):
        self.critic_optim = Adam(self.critic.parameters(), lr=self.config['hp']['lr']['critic'])
        self.motivator_optim = Adam(self.motivator.parameters(), lr=self.config['hp']['lr']['motivator'])
        ae_params = chain(self.encoder.parameters(), self.decoder.parameters(), self.merge_nn.parameters())
        self.ae_optim = Adam(ae_params, lr=self.config['hp']['lr']['ae'])

    def train_on_batch(self, batch):
        rec_loss, motivator_loss, critic_loss, ae_loss = self.loss_on_batch(batch)

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.ae_optim.zero_grad()
        self.motivator_optim.zero_grad()
        motivator_loss.backward(retain_graph=True)
        self.motivator_optim.step()
        self.ae_optim.step()

        self.ae_optim.zero_grad()
        ae_loss.backward(retain_graph=True)
        self.ae_optim.step()

        self.writer.add_scalar('Rec loss', rec_loss, self.num_iters_done)
        self.writer.add_scalar('Motivator loss', motivator_loss, self.num_iters_done)
        self.writer.add_scalar('Critic loss', critic_loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        # Computing codes we need
        style_modern, content_modern = self.encoder(batch.modern)
        style_original, content_original = self.encoder(batch.original)

        # Now we should merge back style and content for decoder
        hid_modern = self.merge_nn(torch.cat([style_modern, content_modern], dim=1))
        hid_original = self.merge_nn(torch.cat([style_original, content_original], dim=1))

        # Reconstructing
        recs_modern = self.decoder(hid_modern, batch.modern[:, :-1])
        recs_original = self.decoder(hid_original, batch.original[:, :-1])

        # Computing reconstruction loss
        rec_loss_modern = self.rec_criterion(recs_modern.view(-1, len(self.vocab)), batch.modern[:, 1:].contiguous().view(-1))
        rec_loss_original = self.rec_criterion(recs_original.view(-1, len(self.vocab)), batch.original[:, 1:].contiguous().view(-1))
        rec_loss = (rec_loss_modern + rec_loss_original) / 2

        # Computing critic loss
        critic_modern_preds, critic_original_preds = self.critic(content_modern), self.critic(content_original)
        critic_loss = self.critic_criterion(critic_modern_preds, critic_original_preds)

        # Computing motivator loss
        motivator_logits_modern = self.motivator(style_modern)
        motivator_logits_original = self.motivator(style_original)
        motivator_loss_modern = self.motivator_criterion(motivator_logits_modern, torch.ones_like(motivator_logits_modern))
        motivator_loss_original = self.motivator_criterion(motivator_logits_original, torch.zeros_like(motivator_logits_original))
        motivator_loss = (motivator_loss_modern + motivator_loss_original) / 2

        # Loss for encoder and decoder is threefold
        coefs = self.config.get('loss_coefs')
        ae_loss = coefs['rec'] * rec_loss + coefs['motivator'] * motivator_loss - coefs['critic'] * critic_loss

        return rec_loss, motivator_loss, critic_loss, ae_loss

    def validate(self):
        rec_losses = []
        motivator_losses = []
        critic_losses = []

        for batch in self.val_dataloader:
            rec_loss, motivator_loss, critic_loss, _ = self.loss_on_batch(batch)
            rec_losses.append(rec_loss.item())
            motivator_losses.append(motivator_loss.item())
            critic_losses.append(critic_loss.item())

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.writer.add_scalar('val_motivator_loss', np.mean(motivator_losses), self.num_iters_done)
        self.writer.add_scalar('val_critic_loss', np.mean(critic_losses), self.num_iters_done)

        self.losses['val_rec_loss'].append(np.mean(rec_losses))

        # Ok, let's now validate style transfer and auto-encoding
        self.validate_inference()

    def validate_inference(self):
        """
        Performs inference on a val dataloader
        (computes predictions without teacher's forcing)
        """
        m2o, o2m, m2m, o2o, gm, go = self.transfer_style(self.val_dataloader)

        m2o_bleu = compute_bleu_for_sents(m2o, go)
        o2m_bleu = compute_bleu_for_sents(o2m, gm)
        m2m_bleu = compute_bleu_for_sents(m2m, gm)
        o2o_bleu = compute_bleu_for_sents(o2o, go)

        self.writer.add_scalar('m2o val BLEU', m2o_bleu, self.num_iters_done)
        self.writer.add_scalar('o2m val BLEU', o2m_bleu, self.num_iters_done)
        self.writer.add_scalar('m2m val BLEU', m2m_bleu, self.num_iters_done)
        self.writer.add_scalar('o2o val BLEU', o2o_bleu, self.num_iters_done)

        # Ok, let's log generated sequences
        texts = [get_text_from_sents(*sents) for sents in zip(m2o, o2m, m2m, o2o, gm, go)]
        text = '\n===================\n'.join(texts)

        self.writer.add_text('Generated examples', text, self.num_iters_done)

    def transfer_style(self, dataloader):
        """
        Produces predictions for a given dataloader
        """
        modern_to_original = []
        original_to_modern = []
        modern_to_modern = []
        original_to_original = []
        gold_modern = []
        gold_original = []

        for batch in dataloader:
            m2o, o2m, m2m, o2o = self.transfer_style_on_batch(batch)

            modern_to_original.extend(m2o)
            original_to_modern.extend(o2m)
            modern_to_modern.extend(m2m)
            original_to_original.extend(o2o)

            gold_modern.extend(batch.modern.detach().cpu().numpy().tolist())
            gold_original.extend(batch.original.detach().cpu().numpy().tolist())

        # Converting to sentences
        m2o_sents = itos_many(modern_to_original, self.vocab)
        o2m_sents = itos_many(original_to_modern, self.vocab)
        m2m_sents = itos_many(modern_to_modern, self.vocab)
        o2o_sents = itos_many(original_to_original, self.vocab)
        gm_sents = itos_many(gold_modern, self.vocab)
        go_sents = itos_many(gold_original, self.vocab)

        return m2o_sents, o2m_sents, m2m_sents, o2o_sents, gm_sents, go_sents

    def transfer_style_on_batch(self, batch):
        style_modern, content_modern = self.encoder(batch.modern)
        style_original, content_original = self.encoder(batch.original)

        modern_to_original_z = self.merge_nn(torch.cat([style_original, content_modern], dim=1))
        original_to_modern_z = self.merge_nn(torch.cat([style_modern, content_original], dim=1))
        modern_to_modern_z = self.merge_nn(torch.cat([style_modern, content_modern], dim=1))
        original_to_original_z = self.merge_nn(torch.cat([style_original, content_original], dim=1))

        m2o = inference(self.decoder, modern_to_original_z, self.vocab)
        o2m = inference(self.decoder, original_to_modern_z, self.vocab)
        m2m = inference(self.decoder, modern_to_modern_z, self.vocab)
        o2o = inference(self.decoder, original_to_original_z, self.vocab)

        return m2o, o2m, m2m, o2o


def get_text_from_sents(m2o_s, o2m_s, m2m_s, o2o_s, gm_s, go_s):
    # TODO: Move this somewhere from trainer file? Or create some nice template?
    return """
        Gold modern: [{}]
        Gold origin: [{}]

        m2o: [{}]
        o2m: [{}]
        m2m: [{}]
        o2o: [{}]
    """.format(gm_s, go_s, m2o_s, o2m_s, m2m_s, o2o_s)
