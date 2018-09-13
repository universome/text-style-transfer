import os
import math
from itertools import chain

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils.training_utils import cudable, grad_norm
from firelab.utils.data_utils import filter_sents_by_len
from sklearn.model_selection import train_test_split

from src.utils.data_utils import itos_many
from src.inference import InferenceState
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.gan_losses import DiscriminatorLoss
from src.models.transformer import TransformerEncoder, TransformerDecoder, TransformerEmbedder, TransformerLM


class LMDiscriminatorsTrainer(BaseTrainer):
    def __init__(self, config):
        super(LMDiscriminatorsTrainer, self).__init__(config)

        self.losses['val_rec_loss'] = [] # Here we'll write history for early stopping

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        domain_x_data_path = os.path.join(project_path, self.config.data.domain_x)
        domain_y_data_path = os.path.join(project_path, self.config.data.domain_y)

        with open(domain_x_data_path) as f: domain_x = f.read().splitlines()
        with open(domain_y_data_path) as f: domain_y = f.read().splitlines()

        domain_x = [s for s in domain_x if self.config.hp.min_len <= len(s.split()) <= self.config.hp.max_len]
        domain_y = [s for s in domain_y if self.config.hp.min_len <= len(s.split()) <= self.config.hp.max_len]

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
        self.encoder = cudable(TransformerEmbedder(self.config.hp.n_vecs, self.config.hp.transformer, self.vocab))
        self.decoder = cudable(TransformerDecoder(self.config.hp.transformer, self.vocab))
        self.lm_x = cudable(TransformerLM(self.config.hp.transformer, self.vocab))
        self.lm_y = cudable(TransformerLM(self.config.hp.transformer, self.vocab))
        # self.x2y = cudable(TransformerEncoder(self.config.hp.transformer, self.vocab))
        # self.y2x = cudable(TransformerEncoder(self.config.hp.transformer, self.vocab))
        self.emb_x = cudable(T.normal(T.zeros(self.config.hp.transformer.d_model)))
        self.emb_y = cudable(T.normal(T.zeros(self.config.hp.transformer.d_model)))

        self.emb_x.requires_grad = True
        self.emb_y.requires_grad = True

    def init_criterions(self):
        self.rec_criterion = cross_entropy_without_pads(self.vocab)
        self.critic_criterion = DiscriminatorLoss()
        self.kl = nn.KLDivLoss(reduction='none')

    def init_optimizers(self):
        self.lm_optim = Adam(chain(self.lm_x.parameters(), self.lm_y.parameters()), lr=self.config.hp.lr)
        self.ae_optim = Adam(chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            [self.emb_x, self.emb_y],
        ), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        ae_loss, lm_loss, losses_info = self.loss_on_batch(batch)

        self.lm_optim.zero_grad()
        lm_loss.backward()

        losses_info['lm_x_grad_norm'] = grad_norm(self.lm_x.parameters()).item()
        losses_info['lm_y_grad_norm'] = grad_norm(self.lm_y.parameters()).item()

        clip_grad_norm_(self.lm_x.parameters(), self.config.hp.grad_clip)
        clip_grad_norm_(self.lm_y.parameters(), self.config.hp.grad_clip)

        self.lm_optim.step()

        self.ae_optim.zero_grad()
        ae_loss.backward()

        losses_info['encoder_grad_norm'] = grad_norm(self.encoder.parameters()).item()
        losses_info['decoder_grad_norm'] = grad_norm(self.decoder.parameters()).item()
        losses_info['emb_x_grad_norm'] = grad_norm([self.emb_x]).item()
        losses_info['emb_y_grad_norm'] = grad_norm([self.emb_y]).item()

        clip_grad_norm_(self.encoder.parameters(), self.config.hp.grad_clip)
        clip_grad_norm_(self.decoder.parameters(), self.config.hp.grad_clip)
        clip_grad_norm_([self.emb_x], self.config.hp.grad_clip)
        clip_grad_norm_([self.emb_y], self.config.hp.grad_clip)
        self.ae_optim.step()

        self.write_losses(losses_info, prefix='TRAIN/')

    def loss_on_batch(self, batch):
        # LM losses
        lm_loss_x = self.lm_loss_on_batch(self.lm_x, batch.domain_x)
        lm_loss_y = self.lm_loss_on_batch(self.lm_y, batch.domain_y)
        lm_loss = (lm_loss_x + lm_loss_y) / 2

        # Reconstruction losses
        rec_loss_x = self.rec_loss_on_batch(batch.domain_x, self.emb_x)
        rec_loss_y = self.rec_loss_on_batch(batch.domain_y, self.emb_y)
        rec_loss = (rec_loss_x + rec_loss_y) / 2

        # Generator adversarial loss
        adv_loss_x2y = self.gen_loss_on_batch(self.lm_y, batch.domain_x, self.emb_y)
        adv_loss_y2x = self.gen_loss_on_batch(self.lm_x, batch.domain_y, self.emb_x)
        adv_loss = (adv_loss_x2y + adv_loss_y2x) / 2

        ae_loss = rec_loss + adv_loss

        losses_info = {
            'rec_loss_x': rec_loss_x,
            'rec_loss_y': rec_loss_y,
            'adv_loss_x2y': adv_loss_x2y,
            'adv_loss_y2x': adv_loss_y2x,
            'lm_loss_x': lm_loss_x,
            'lm_loss_y': lm_loss_y,
        }

        return ae_loss, lm_loss, losses_info

    def lm_loss_on_batch(self, lm, text):
        preds = lm(text[:, :-1])
        loss = self.rec_criterion(preds.view(-1, len(self.vocab)), text[:, 1:].contiguous().view(-1))

        return loss

    def rec_loss_on_batch(self, text, style_emb):
        embs = self.encoder(text)
        embs = T.cat((embs, style_emb.repeat(embs.size(0), 1, 1)), dim=1)
        recs = self.decoder(embs, text[:, :-1], None)
        loss = self.rec_criterion(recs.view(-1, len(self.vocab)), text[:, 1:].contiguous().view(-1))

        return loss

    def gen_loss_on_batch(self, lm, text, style_emb):
        # We should generate sentence with gumbel
        # Then pass it to LM to get domain predictions
        # Our loss is KL loss between LM predictions and decoder predictions
        embs = self.encoder(text)
        embs = T.cat((embs, style_emb.repeat(embs.size(0), 1, 1)), dim=1)

        lens = [s.cpu().numpy().tolist().index(self.vocab.stoi['<eos>']) + 1 for s in text]
        preds = InferenceState({
            'model': self.decoder,
            'gumbel': True,
            'inputs': embs,
            'vocab': self.vocab,
            'should_stack_finished': True,
            'max_len': max(lens) # TODO: put back individual max lens!
        })

        lm_preds = lm(preds[:, :-1], onehot=False)
        lm_preds = F.softmax(lm_preds, dim=2)
        losses = self.kl(preds[:, 1:].log(), lm_preds).sum(dim=2)
        losses = losses / cudable(T.tensor(lens).float().unsqueeze(1))
        loss = losses.sum()

        assert not T.isnan(embs).any()
        assert not T.isnan(preds).any()
        assert not T.isnan(lm_preds).any()
        assert not T.isnan(losses).any()

        return loss
