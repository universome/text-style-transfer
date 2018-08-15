import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils import cudable, grad_norm
from sklearn.model_selection import train_test_split

from src.models.transformer.utils import pad_mask
from src.models.transformer import TransformerEncoder, TransformerCritic
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.gan_losses import WCriticLoss, WGeneratorLoss


class WordReplacerTrainer(BaseTrainer):
    def __init__(self, config):
        super(WordReplacerTrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        min_len = self.config.hp.transformer.min_len
        max_len = self.config.hp.transformer.max_len
        project_path = self.config.firelab.project_path
        domain_x_data_path = os.path.join(project_path, self.config.data.domain_x)
        domain_y_data_path = os.path.join(project_path, self.config.data.domain_y)

        with open(domain_x_data_path) as f: domain_x = f.read().splitlines()
        with open(domain_y_data_path) as f: domain_y = f.read().splitlines()

        domain_x = [s for s in domain_x if min_len <= len(s.split()) <= max_len-3]
        domain_y = [s for s in domain_y if min_len <= len(s.split()) <= max_len-3]

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
        self.generator_x2y = cudable(nn.ModuleList([
            TransformerEncoder(self.config.hp.transformer, self.vocab),
            nn.Linear(self.config.hp.transformer.d_model, len(self.vocab))
        ]))
        self.critic_y = cudable(TransformerCritic(self.config.hp.transformer, self.vocab))

    def init_criterions(self):
        self.gen_criterion = WGeneratorLoss()
        self.critic_criterion = WCriticLoss()

    def init_optimizers(self):
        self.optim_gen = Adam(self.generator_x2y.parameters(), lr=self.config.hp.lr)
        self.optim_critic = Adam(self.critic_y.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        # Try training crtici for N steps and generator for 1 step
        # Try reconstruction loss for generator (gradually reducing it)
        gen_loss, critic_loss, bleu, *_ = self.loss_on_batch(batch)

        self.optim_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optim_critic.step()

        self.optim_gen.zero_grad()
        gen_loss.backward()
        self.optim_gen.step()

        self.writer.add_scalar('Train/critic_loss', critic_loss.item(), self.num_iters_done)
        self.writer.add_scalar('Train/gen_loss', gen_loss.item(), self.num_iters_done)
        self.writer.add_scalar('Train/bleu', bleu, self.num_iters_done)

    def loss_on_batch(self, batch):
        embs, _ = self.generator_x2y[0](batch.domain_x)
        logits = self.generator_x2y[1](embs)
        enc_mask = pad_mask(batch.domain_x, self.vocab).unsqueeze(1)

        preds_on_fake = self.critic_y(logits, enc_mask, onehot=False)
        preds_on_real = self.critic_y(batch.domain_y)

        gen_loss = self.gen_criterion(preds_on_fake)
        critic_loss = self.critic_criterion(preds_on_real, preds_on_fake)

        tokens = logits.max(dim=-1)[1]

        x = itos_many(batch.domain_x, self.vocab)
        x2y = itos_many(tokens, self.vocab)
        bleu = compute_bleu_for_sents(x2y, x)

        return gen_loss, critic_loss, bleu, x, x2y

    def validate(self):
        gen_losses = []
        critic_losses = []
        bleus = []
        x_all = []
        x2y_all = []

        for batch in self.val_dataloader:
            gen_loss, critic_loss, bleu, x, x2y = self.loss_on_batch(batch)
            gen_losses.append(gen_loss.item())
            critic_losses.append(critic_loss.item())
            bleus.append(bleu)
            x_all.extend(x)
            x2y_all.extend(x2y)

        self.writer.add_scalar('VAL/gen', np.mean(gen_losses), self.num_iters_done)
        self.writer.add_scalar('VAL/critic', np.mean(critic_losses), self.num_iters_done)
        self.writer.add_scalar('VAL/bleu', np.mean(bleus), self.num_iters_done)

        texts = ['{} => {}'.format(s_x, s_x2y) for s_x, s_x2y in zip(x_all, x2y_all)]
        texts = texts[:10] # If we'll write too many texts, nothing will be displayed in TB
        text = '\n ---------- \n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
