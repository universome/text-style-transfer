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
from firelab.utils.training_utils import cudable, grad_norm
from sklearn.model_selection import train_test_split

from src.models.dissonet import MergeNN, SplitNN, DissoNet
from src.models import FFN, RNNEncoder, RNNDecoder
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.gan_losses import DiscriminatorLoss
from src.inference import simple_inference
from src.utils.style_transfer import transfer_style, get_text_from_sents
from src.utils.data_utils import char_tokenize

class DissoNetTrainer(BaseTrainer):
    def __init__(self, config):
        super(DissoNetTrainer, self).__init__(config)

        self.losses['val_rec_loss'] = [] # Here we'll write history for early stopping

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        domain_x_data_path = os.path.join(project_path, self.config.data.domain_x)
        domain_y_data_path = os.path.join(project_path, self.config.data.domain_y)

        with open(domain_x_data_path) as f: domain_x = f.read().splitlines()
        with open(domain_y_data_path) as f: domain_y = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='|', batch_first=True, tokenize=char_tokenize)
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
        size = self.config.hp.size
        dropout_p = self.config.hp.dropout
        dropword_p = self.config.hp.dropword

        self.encoder = RNNEncoder(size, size, self.vocab, dropword_p)
        self.decoder = RNNDecoder(size, size, self.vocab, dropword_p)
        self.split_nn = SplitNN(size, self.config.hp.style_vec_size)
        self.motivator = FFN([self.config.hp.style_vec_size, 1], dropout=dropout_p)
        self.critic = FFN([size, size, 1], dropout=dropout_p)
        self.merge_nn = MergeNN(size, self.config.hp.style_vec_size)

        # Let's save all ae params into single list for future use
        self.ae_params = list(chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.split_nn.parameters(),
            self.merge_nn.parameters(),
        ))

        self.dissonet = cudable(DissoNet(self.encoder, self.decoder, self.split_nn, self.motivator, self.critic, self.merge_nn))

        if torch.cuda.device_count() > 1:
            print('Going to parallelize on {} GPUs'.format(torch.cuda.device_count()))
            self.dissonet = nn.DataParallel(self.dissonet)

    def init_criterions(self):
        self.rec_criterion = cross_entropy_without_pads(self.vocab)
        self.critic_criterion = DiscriminatorLoss()
        self.motivator_criterion = nn.BCEWithLogitsLoss()

    def init_optimizers(self):
        self.ae_optim = Adam(self.ae_params, lr=self.config.hp.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.config.hp.lr)
        self.motivator_optim = Adam(self.motivator.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        if batch.batch_size == 1: return # We can't train on batches of size 1 because of BatchNorm

        ae_loss, motivator_loss, critic_loss, losses_info = self.loss_on_batch(batch)

        self.ae_optim.zero_grad()
        ae_loss.backward(retain_graph=True)
        ae_grad_norm = grad_norm(self.ae_params)
        clip_grad_norm_(self.ae_params, self.config.hp.grad_clip)
        self.ae_optim.step()

        self.motivator_optim.zero_grad()
        motivator_loss.backward(retain_graph=True)
        ae_grad_norm = grad_norm(self.motivator.parameters())
        clip_grad_norm_(self.motivator.parameters(), self.config.hp.grad_clip)
        self.motivator_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = grad_norm(self.critic.parameters())
        clip_grad_norm_(self.critic.parameters(), self.config.hp.grad_clip)
        self.critic_optim.step()

        self.write_losses(losses_info, prefix='TRAIN/')
        self.writer.add_scalar('TRAIN/grad_norm_ae', ae_grad_norm, self.num_iters_done)
        self.writer.add_scalar('TRAIN/grad_norm_critic', critic_grad_norm, self.num_iters_done)

    def loss_on_batch(self, batch):
        recs_x, recs_y, critic_preds_x, critic_preds_y, motivator_preds_x, motivator_preds_y, _, _ = self.dissonet(batch.domain_x, batch.domain_y)

        # Critic loss
        critic_loss = self.critic_criterion(critic_preds_x, critic_preds_y)

        # Computing reconstruction loss
        rec_loss_x = self.rec_criterion(recs_x.view(-1, len(self.vocab)), batch.domain_x[:, 1:].contiguous().view(-1))
        rec_loss_y = self.rec_criterion(recs_y.view(-1, len(self.vocab)), batch.domain_y[:, 1:].contiguous().view(-1))
        rec_loss = (rec_loss_x + rec_loss_y) / 2

        # Motivator loss
        motivator_loss_x = self.motivator_criterion(motivator_preds_x, torch.ones_like(motivator_preds_x))
        motivator_loss_y = self.motivator_criterion(motivator_preds_y, torch.ones_like(motivator_preds_y))
        motivator_loss = (motivator_loss_x + motivator_loss_y) / 2

        # AE loss is twofold
        critic_coef = 0 if critic_loss.item() > self.config.hp.critic_loss_threshold else 1
        motivator_coef = self.config.hp.motivator_coef
        ae_loss = rec_loss - critic_coef * critic_loss + motivator_coef * motivator_loss

        losses_info = {
            'rec_loss_x': rec_loss_x.item(),
            'rec_loss_y': rec_loss_y.item(),
            'motivator_loss_x': motivator_loss_x.item(),
            'motivator_loss_y': motivator_loss_y.item(),
            'critic_loss': critic_loss.item(),
        }

        return ae_loss, motivator_loss, critic_loss, losses_info

    def validate(self):
        losses = []
        rec_losses = []

        for batch in self.val_dataloader:
            batch = cudable(batch)
            with torch.enable_grad():
                rec_loss, *_, losses_info = self.loss_on_batch(batch)
            rec_losses.append(rec_loss.item())
            losses.append(losses_info)

        for l in losses[0].keys():
            value = np.mean([info[l] for info in losses])
            self.writer.add_scalar('VAL/' + l, value, self.num_iters_done)

        self.losses['val_rec_loss'].append(np.mean(rec_losses))

        # Ok, let's now validate style transfer and auto-encoding
        self.validate_inference()

    def validate_inference(self):
        """
        Performs inference on a val dataloader
        (computes predictions without teacher's forcing)
        """
        x2y, y2x, x2x, y2y, gx, gy = transfer_style(self.transfer_style_on_batch, self.val_dataloader, self.vocab, sep='')

        x2y_bleu = compute_bleu_for_sents(x2y, gx)
        y2x_bleu = compute_bleu_for_sents(y2x, gy)
        x2x_bleu = compute_bleu_for_sents(x2x, gx)
        y2y_bleu = compute_bleu_for_sents(y2y, gy)

        self.writer.add_scalar('VAL/BLEU/x2y', x2y_bleu, self.num_iters_done)
        self.writer.add_scalar('VAL/BLEU/y2x', y2x_bleu, self.num_iters_done)
        self.writer.add_scalar('VAL/BLEU/x2x', x2x_bleu, self.num_iters_done)
        self.writer.add_scalar('VAL/BLEU/y2y', y2y_bleu, self.num_iters_done)

        # Ok, let's log generated sequences
        texts = [get_text_from_sents(*sents) for sents in zip(x2y, y2x, x2x, y2y, gx, gy)]
        texts = texts[:10] # If we'll write too many texts, nothing will be displayed in TB
        text = '\n===================\n'.join(texts)

        self.writer.add_text('Generated examples', text, self.num_iters_done)

    def transfer_style_on_batch(self, batch):
        batch = cudable(batch)

        state_x = self.encoder(batch.domain_x)
        state_y = self.encoder(batch.domain_y)

        content_x, style_x = self.dissonet.split_nn(state_x)
        content_y, style_y = self.dissonet.split_nn(state_y)

        state_x2y = self.merge_nn(content_x, style_y)
        state_y2x = self.merge_nn(content_y, style_x)
        state_x2x = self.merge_nn(content_x, style_x)
        state_y2y = self.merge_nn(content_y, style_y)

        x2y = simple_inference(self.decoder, state_x2y, self.vocab, eos_token='|')
        y2x = simple_inference(self.decoder, state_y2x, self.vocab, eos_token='|')
        x2x = simple_inference(self.decoder, state_x2x, self.vocab, eos_token='|')
        y2y = simple_inference(self.decoder, state_y2y, self.vocab, eos_token='|')

        return x2y, y2x, x2x, y2y
