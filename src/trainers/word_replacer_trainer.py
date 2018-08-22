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
from firelab.utils import cudable, grad_norm, determine_turn, HPLinearScheme, onehot_encode
from sklearn.model_selection import train_test_split

from src.models.transformer.utils import pad_mask
from src.models.transformer import TransformerEncoder, TransformerCritic
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.gan_losses import WCriticLoss, WGeneratorLoss, wgan_gp


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
            nn.Linear(self.config.hp.transformer.d_model, len(self.vocab)),
            nn.Softmax(dim=2)
        ]))
        self.critic_y = cudable(TransformerCritic(self.config.hp.transformer, self.vocab))

        if self.config.hp.share_embeddings:
            self.generator_x2y[0].embed.weight = self.critic_y.encoder.embed.weight

    def init_criterions(self):
        self.rec_crit = nn.CrossEntropyLoss() # Try label smoothing!
        self.gen_criterion = WGeneratorLoss()
        self.critic_criterion = WCriticLoss()

    def init_optimizers(self):
        self.optim_gen = Adam(self.generator_x2y.parameters(), lr=self.config.hp.lr)
        self.optim_critic = Adam(self.critic_y.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        gen_total_loss, critic_loss, losses_info, *_ = self.loss_on_batch(batch)

        if determine_turn(self.num_iters_done, self.config.hp.gan_sequencing) == 0:
            self.optim_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.optim_critic.step()
        else:
            self.optim_gen.zero_grad()
            gen_total_loss.backward()
            self.optim_gen.step()

        self.write_losses({('TRAIN/' + k): losses_info[k] for k in losses_info})

    def loss_on_batch(self, batch):
        embs, _ = self.generator_x2y[0](batch.domain_x)
        logits = self.generator_x2y[1](embs)
        probs = self.generator_x2y[2](logits)
        enc_mask = pad_mask(batch.domain_x, self.vocab).unsqueeze(1)

        preds_on_fake = self.critic_y(probs, enc_mask, onehot=False)
        preds_on_real = self.critic_y(batch.domain_y)

        rec_loss = self.rec_crit(logits.view(-1, len(self.vocab)), batch.domain_x.contiguous().view(-1))
        gen_loss = self.gen_criterion(preds_on_fake)
        rec_loss_coef = HPLinearScheme(*self.config.hp.loss_coefs.rec).evaluate(self.num_iters_done)
        gen_loss_coef = HPLinearScheme(*self.config.hp.loss_coefs.gen).evaluate(self.num_iters_done)
        gen_total_loss = rec_loss_coef * rec_loss + gen_loss_coef * gen_loss

        true_onehot = onehot_encode(batch.domain_x, len(self.vocab)).float()
        gp = wgan_gp(self.critic_y, true_onehot, probs, enc_mask, onehot=False)
        critic_loss = self.critic_criterion(preds_on_real, preds_on_fake)
        critic_total_loss = critic_loss + self.config.hp.gp_lambda * gp

        tokens = logits.max(dim=2)[1]

        x = itos_many(batch.domain_x, self.vocab)
        x2y = itos_many(tokens, self.vocab)
        bleu = compute_bleu_for_sents(x2y, x)

        losses_info = {
            'rec_loss': rec_loss.item(),
            'gen_loss': gen_loss.item(),
            'critic_loss': critic_loss.item(),
            'bleu': bleu,
            'gp': gp.item(),
        }

        return gen_total_loss, critic_total_loss, losses_info, x, x2y

    def validate(self):
        losses = []
        x_all = []
        x2y_all = []

        for batch in self.val_dataloader:
            with torch.enable_grad():
                _, _, losses_info, x, x2y = self.loss_on_batch(batch)
            losses.append(losses_info)
            x_all.extend(x)
            x2y_all.extend(x2y)

        for l in losses[0].keys():
            value = np.mean([info[l] for info in losses])
            self.writer.add_scalar('VAL/' + l, value, self.num_iters_done)

        texts = ['Source: {} \n\n Result: {}'.format(s_x, s_x2y) for s_x, s_x2y in zip(x_all, x2y_all)]
        texts = texts[:10] # If we'll write too many texts, nothing will be displayed in TB
        text = '\n ---------- \n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
