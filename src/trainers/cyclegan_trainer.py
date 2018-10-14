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
from firelab.utils.training_utils import cudable, grad_norm, determine_turn
from sklearn.model_selection import train_test_split

from src.models import FFN, RNNEncoder, RNNDecoder
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.gan_losses import WCriticLoss, WGeneratorLoss, wgan_gp
from src.inference import simple_inference
from src.utils.style_transfer import transfer_style, get_text_from_sents
from src.utils.data_utils import char_tokenize
from src.seq_noise import seq_noise


class CycleGANTrainer(BaseTrainer):
    def __init__(self, config):
        super(CycleGANTrainer, self).__init__(config)

    def init_dataloaders(self):
        project_path = self.config.firelab.project_path
        domain_x_data_path = os.path.join(project_path, self.config.data.domain_x)
        domain_y_data_path = os.path.join(project_path, self.config.data.domain_y)

        with open(domain_x_data_path) as f: domain_x = f.read().splitlines()
        with open(domain_y_data_path) as f: domain_y = f.read().splitlines()

        print('Dataset sizes:', len(domain_x), len(domain_y))
        domain_x = [s for s in domain_x if self.config.hp.min_len <= len(s) <= self.config.hp.max_len]
        domain_y = [s for s in domain_y if self.config.hp.min_len <= len(s) <= self.config.hp.max_len]
        print('Dataset sizes after filtering:', len(domain_x), len(domain_y))

        field = Field(init_token='<bos>', eos_token='|', batch_first=True, tokenize=char_tokenize)
        fields = [('domain_x', field), ('domain_y', field)]

        examples = [Example.fromlist([x,y,x], fields) for x,y in zip(domain_x, domain_y)]
        train_exs, val_exs = train_test_split(examples, test_size=self.config.val_set_size,
                                              random_state=self.config.random_seed)

        train_ds, val_ds = Dataset(train_exs, fields), Dataset(val_exs, fields)
        field.build_vocab(train_ds, max_size=self.config.hp.get('max_vocab_size'))

        self.vocab = field.vocab
        self.train_dataloader = data.BucketIterator(train_ds, self.config.hp.batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(val_ds, self.config.hp.batch_size, repeat=False, shuffle=False)

    def init_models(self):
        size = self.config.hp.model_size
        dropout_p = self.config.hp.dropout
        dropword_p = self.config.hp.dropword

        self.encoder = cudable(RNNEncoder(size, size, self.vocab, dropword_p, noise=self.config.hp.noiseness))
        self.decoder = cudable(RNNDecoder(size, size, self.vocab, dropword_p))

        def create_critic():
            return FFN([size, size, 1], dropout_p)

        # GAN from X to Y
        self.gen_x2y = cudable(Generator(size, self.config.hp.gen_n_rec_steps))
        self.critic_y = cudable(create_critic())

        # GAN from Y to X
        self.gen_y2x = cudable(Generator(size, self.config.hp.gen_n_rec_steps))
        self.critic_x = cudable(create_critic())

    def init_criterions(self):
        self.rec_criterion = cross_entropy_without_pads(self.vocab)
        self.critic_criterion = WCriticLoss()
        self.generator_criterion = WGeneratorLoss()

    def init_optimizers(self):
        self.ae_params = list(chain(self.encoder.parameters(), self.decoder.parameters()))
        self.gen_params = list(chain(self.gen_x2y.parameters(), self.gen_y2x.parameters()))
        self.critics_params = list(chain(self.critic_x.parameters(), self.critic_y.parameters()))

        self.ae_optim = Adam(self.ae_params, lr=self.config.hp.lr)
        self.gen_optim = Adam(chain(self.gen_params, self.encoder.parameters()), lr=self.config.hp.lr)
        self.critic_optim = Adam(self.critics_params,
            lr=self.config.hp.critic_optim.lr, betas=self.config.hp.critic_optim.betas)

    def train_on_batch(self, batch):
        if self.num_epochs_done < self.config.hp.ae_pretraining_n_epochs:
            self.train_ae_on_batch(batch)
        else:
            self.train_gans_on_batch(batch)

    def train_gans_on_batch(self, batch):
        if determine_turn(self.num_iters_done, [self.config.hp.n_critic, 1]) == 0:
            self.train_critic_on_batch(batch)
        else:
            self.train_gen_on_batch(batch)

    def train_ae_on_batch(self, batch):
        rec_loss, losses_info = self.ae_loss_on_batch(batch)

        # AE
        self.ae_optim.zero_grad()
        rec_loss.backward()
        ae_grad_norm = grad_norm(self.ae_params)
        clip_grad_norm_(self.ae_params, self.config.hp.grad_clip)
        self.ae_optim.step()

        losses_info['grad norm/ae'] = ae_grad_norm.item()
        self.write_losses(losses_info, prefix='TRAIN/')

    def train_critic_on_batch(self, batch):
        critics_loss, losses_info = self.critic_loss_on_batch(batch)

        # Critics
        self.critic_optim.zero_grad()
        critics_loss.backward(retain_graph=True)
        critic_grad_norm = grad_norm(self.critics_params)
        clip_grad_norm_(self.critics_params, self.config.hp.grad_clip)
        self.critic_optim.step()

        losses_info['grad norm/critic'] = critic_grad_norm.item()
        self.write_losses(losses_info, prefix='TRAIN/')

    def train_gen_on_batch(self, batch):
        ae_loss, ae_losses_info = self.ae_loss_on_batch(batch) # Never stop trainig AE
        gen_loss, gen_losses_info = self.gen_loss_on_batch(batch)

        # Generators
        self.gen_optim.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_grad_norm = grad_norm(self.gen_params)
        clip_grad_norm_(self.gen_params, self.config.hp.grad_clip)
        self.gen_optim.step()

        # AE
        self.ae_optim.zero_grad()
        ae_loss.backward(retain_graph=True)
        ae_grad_norm = grad_norm(self.ae_params)
        clip_grad_norm_(self.ae_params, self.config.hp.grad_clip)
        self.ae_optim.step()

        gen_losses_info['grad norm/gen'] = gen_grad_norm.item()
        ae_losses_info['grad norm/ae'] = ae_grad_norm.item()
        self.write_losses(gen_losses_info, prefix='TRAIN/')
        self.write_losses(ae_losses_info, prefix='TRAIN/')

    def ae_loss_on_batch(self, batch):
        batch.domain_x = cudable(batch.domain_x)
        batch.domain_y = cudable(batch.domain_y)

        x_hid = self.encoder(batch.domain_x)
        y_hid = self.encoder(batch.domain_y)

        # Reconstruction loss
        recs_x = self.decoder(x_hid, batch.domain_x[:, :-1])
        recs_y = self.decoder(y_hid, batch.domain_y[:, :-1])
        rec_loss_x = self.rec_criterion(recs_x.view(-1, len(self.vocab)), batch.domain_x[:, 1:].contiguous().view(-1))
        rec_loss_y = self.rec_criterion(recs_y.view(-1, len(self.vocab)), batch.domain_y[:, 1:].contiguous().view(-1))
        rec_loss = (rec_loss_x + rec_loss_y) / 2

        losses_info = {
            'rec_loss/domain_x': rec_loss_x.item(),
            'rec_loss/domain_y': rec_loss_y.item(),
        }

        return rec_loss, losses_info

    def critic_loss_on_batch(self, batch):
        batch.domain_x = cudable(batch.domain_x)
        batch.domain_y = cudable(batch.domain_y)

        x_hid = self.encoder(batch.domain_x)
        y_hid = self.encoder(batch.domain_y)
        x2y_hid = self.gen_x2y(x_hid)
        y2x_hid = self.gen_y2x(y_hid)

        # Critic loss
        critic_x_preds_x, critic_x_preds_y2x = self.critic_x(x_hid), self.critic_x(y2x_hid)
        critic_y_preds_y, critic_y_preds_x2y = self.critic_y(y_hid), self.critic_y(x2y_hid)
        critic_x_loss = self.critic_criterion(critic_x_preds_x, critic_x_preds_y2x)
        critic_y_loss = self.critic_criterion(critic_y_preds_y, critic_y_preds_x2y)
        critic_x_gp = wgan_gp(self.critic_x, x_hid, y2x_hid)
        critic_y_gp = wgan_gp(self.critic_y, y_hid, x2y_hid)
        critic_x_total_loss = critic_x_loss + self.config.hp.gp_lambda * critic_x_gp
        critic_y_total_loss = critic_y_loss + self.config.hp.gp_lambda * critic_y_gp
        critics_total_loss = (critic_x_total_loss + critic_y_total_loss) / 2

        losses_info = {
            'critic_loss/domain_x': critic_x_loss.item(),
            'critic_loss/domain_y': critic_y_loss.item(),
            'critic_loss/gp_x': critic_x_gp.item(),
            'critic_loss/gp_y': critic_y_gp.item(),
        }

        return critics_total_loss, losses_info

    def gen_loss_on_batch(self, batch):
        batch.domain_x = cudable(batch.domain_x)
        batch.domain_y = cudable(batch.domain_y)

        x_hid = self.encoder(batch.domain_x)
        y_hid = self.encoder(batch.domain_y)
        x2y_hid = self.gen_x2y(x_hid)
        y2x_hid = self.gen_y2x(y_hid)
        x2y2x_hid = self.gen_y2x(x2y_hid)
        y2x2y_hid = self.gen_x2y(y2x_hid)

        # Generator loss (consists of making critic's life harder and lp loss)
        critic_x_preds_y2x = self.critic_x(y2x_hid)
        critic_y_preds_x2y = self.critic_y(x2y_hid)
        adv_x_loss = self.generator_criterion(critic_y_preds_x2y)
        adv_y_loss = self.generator_criterion(critic_x_preds_y2x)
        adv_loss = (adv_x_loss + adv_y_loss) / 2

        x_hid_lp_loss = torch.norm(x_hid - x2y2x_hid, p=self.config.hp.p_norm)
        y_hid_lp_loss = torch.norm(y_hid - y2x2y_hid, p=self.config.hp.p_norm)
        lp_loss = (x_hid_lp_loss + y_hid_lp_loss) / 2

        # Total loss
        gen_loss = adv_loss + self.config.hp.lp_loss_coef * lp_loss

        losses_info = {
            'gen_adv_loss/domain_x': adv_x_loss.item(),
            'gen_adv_loss/domain_y': adv_y_loss.item(),
            'gen_lp_loss/domain_x': x_hid_lp_loss.item(),
            'gen_lp_loss/domain_y': y_hid_lp_loss.item(),
        }

        return gen_loss, losses_info

    def validate(self):
        losses = []

        for batch in self.val_dataloader:
            batch.domain_x = cudable(batch.domain_x)
            batch.domain_y = cudable(batch.domain_y)

            *_, rec_losses_info = self.ae_loss_on_batch(batch)
            *_, gen_losses_info = self.gen_loss_on_batch(batch)
            with torch.enable_grad():
                *_, critic_losses_info = self.critic_loss_on_batch(batch)
            losses_info = dict(list(critic_losses_info.items()) + list(gen_losses_info.items()) + list(rec_losses_info.items()))
            losses.append(losses_info)

        for l in losses[0].keys():
            value = np.mean([info[l] for info in losses])
            self.writer.add_scalar('VAL/' + l, value, self.num_iters_done)

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

        self.writer.add_scalar('VAL_BLEU/x2y', x2y_bleu, self.num_iters_done)
        self.writer.add_scalar('VAL_BLEU/y2x', y2x_bleu, self.num_iters_done)
        self.writer.add_scalar('VAL_BLEU/x2x', x2x_bleu, self.num_iters_done)
        self.writer.add_scalar('VAL_BLEU/y2y', y2y_bleu, self.num_iters_done)

        # Ok, let's log generated sequences
        texts = [get_text_from_sents(*sents) for sents in zip(x2y, y2x, x2x, y2y, gx, gy)]
        texts = texts[:10] # Let's not display all the texts
        text = '\n===================\n'.join(texts)

        self.writer.add_text('Generated examples', text, self.num_iters_done)

    def transfer_style_on_batch(self, batch):
        batch.domain_x = cudable(batch.domain_x)
        batch.domain_y = cudable(batch.domain_y)

        x_z = self.encoder(batch.domain_x)
        y_z = self.encoder(batch.domain_y)

        x2y_z = self.gen_x2y(x_z)
        y2x_z = self.gen_y2x(y_z)
        x2x_z = self.gen_y2x(x2y_z)
        y2y_z = self.gen_x2y(y2x_z)

        x2y = simple_inference(self.decoder, x2y_z, self.vocab, eos_token='|')
        y2x = simple_inference(self.decoder, y2x_z, self.vocab, eos_token='|')
        x2x = simple_inference(self.decoder, x2x_z, self.vocab, eos_token='|')
        y2y = simple_inference(self.decoder, y2y_z, self.vocab, eos_token='|')

        return x2y, y2x, x2x, y2y


class Generator(nn.Module):
    def __init__(self, size, n_steps):
        super(Generator, self).__init__()

        self.n_steps = n_steps
        self.gru = nn.GRU(size, size, batch_first=True)

    def forward(self, z):
        x = z.unsqueeze(1).repeat(1, self.n_steps, 1)
        y = self.gru(x, z.unsqueeze(0))[1].squeeze(0)

        return y
