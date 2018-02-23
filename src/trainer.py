from copy import deepcopy

import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.data_utils import pad_to_longest
import src.transformer.constants as constants

use_cuda = torch.cuda.is_available()

LOSSES_TITLES = {
    'dae_loss_src': '[src] lang AE loss',
    'dae_loss_trg': '[trg] lang AE loss',
    'loss_bt_src': '[src] lang back-translation loss',
    'loss_bt_trg': '[trg] lang back-translation loss',
    'discr_loss_src': '[src] lang discriminator loss',
    'discr_loss_trg': '[trg] lang discriminator loss',
    'gen_loss_src': '[src] lang generator loss',
    'gen_loss_trg': '[trg] lang generator loss',
    'src_to_trg_translation': '[src->trg] translation loss',
    'trg_to_src_translation': '[trg->src] translation loss'
}


class Trainer:
    def __init__(self, translator, discriminator, translator_optimizer, discriminator_optimizer, transformer_bt_optimizer,
                reconstruct_src_criterion, reconstruct_trg_criterion, adv_criterion, config):

        self.transformer = translator
        self.discriminator = discriminator
        self.transformer_optimizer = translator_optimizer
        self.transformer_bt_optimizer = transformer_bt_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.reconstruct_src_criterion = reconstruct_src_criterion
        self.reconstruct_trg_criterion = reconstruct_trg_criterion
        self.adv_criterion = adv_criterion

        if use_cuda:
            self.transformer.cuda()
            self.discriminator.cuda()
            self.transformer.cuda()
            self.discriminator.cuda()
            self.reconstruct_src_criterion.cuda()
            self.reconstruct_trg_criterion.cuda()
            self.adv_criterion.cuda()

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 100)
        self.start_bt_from_iter = config.get('start_bt_from_iter', 500)
        self.max_seq_len = config.get('max_seq_len', 50)

        self.losses = {
            'dae_loss_src': [],
            'dae_loss_trg': [],
            'loss_bt_src': [],
            'loss_bt_trg': [],
            'discr_loss_src': [],
            'discr_loss_trg': [],
            'gen_loss_src': [],
            'gen_loss_trg': [],
            'src_to_trg_translation': [],
            'trg_to_src_translation': []
        }

        # Val losses have the same structure, so let's just clone them
        self.val_losses = deepcopy(self.losses)
        self.val_iters = deepcopy(self.losses)

    def run_training(self, training_data, val_data, translation_val_data,
                     plot_every=50, val_translate_every=100):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            self.validate(val_data)

            for batch in tqdm(training_data, leave=False):
                try:
                    self.train_on_batch(batch)
                    if self.num_iters_done % val_translate_every == 0: self.validate_translation(translation_val_data)
                    if self.num_iters_done % plot_every == 0: self.plot_losses()
                    self.num_iters_done += 1
                except KeyboardInterrupt:
                    should_continue = False
                    break

            self.num_epochs_done += 1

    def train_on_batch(self, batch):
        src_noised, trg_noised, src, trg = batch

        # Resetting gradients
        self.transformer_optimizer.zero_grad()
        self.transformer_bt_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        should_backtranslate = self.num_iters_done >= self.start_bt_from_iter

        # DAE step
        self.transformer.train()
        losses, encodings = self._run_dae(src_noised, trg_noised, src, trg)
        dae_loss_src, dae_loss_trg = losses
        dae_loss_src.backward(retain_graph=True)
        dae_loss_trg.backward(retain_graph=True)
        self.transformer_optimizer.step()
        self.losses['dae_loss_src'].append(dae_loss_src.data[0])
        self.losses['dae_loss_trg'].append(dae_loss_trg.data[0])

        # (Back) translation step
        if should_backtranslate:
            loss_bt_src, loss_bt_trg = self._run_bt(src, trg)
            loss_bt_src.backward(retain_graph=True)
            loss_bt_trg.backward(retain_graph=True)
            self.transformer_bt_optimizer.step()
            self.losses['loss_bt_src'].append(loss_bt_src.data[0])
            self.losses['loss_bt_trg'].append(loss_bt_trg.data[0])

        # Discriminator step
        self.transformer_optimizer.zero_grad()
        losses, domains_predictions = self._run_discriminator(*encodings)
        discr_loss_src, discr_loss_trg = losses
        discr_loss_src.backward(retain_graph=True)
        discr_loss_trg.backward(retain_graph=True)
        self.discriminator_optimizer.step()
        self.losses['discr_loss_src'].append(discr_loss_src.data[0])
        self.losses['discr_loss_trg'].append(discr_loss_trg.data[0])

        # Generator step
        self.transformer_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        gen_loss_src, gen_loss_trg = self._run_generator(*domains_predictions)
        gen_loss_src.backward(retain_graph=True)
        gen_loss_trg.backward(retain_graph=True)
        self.transformer_optimizer.step()
        self.losses['gen_loss_src'].append(gen_loss_src.data[0])
        self.losses['gen_loss_trg'].append(gen_loss_trg.data[0])

    def eval_on_batch(self, batch):
        src_noised, trg_noised, src, trg = batch
        should_backtranslate = self.num_iters_done >= self.start_bt_from_iter

        self.transformer.eval()
        self.discriminator.eval()

        # DAE step
        losses, encodings = self._run_dae(src_noised, trg_noised, src, trg)
        dae_loss_src, dae_loss_trg = losses
        self.val_losses['dae_loss_src'].append(dae_loss_src.data[0])
        self.val_losses['dae_loss_trg'].append(dae_loss_trg.data[0])
        self.val_iters['dae_loss_src'].append(self.num_iters_done)
        self.val_iters['dae_loss_trg'].append(self.num_iters_done)

        # (Back) translation step
        if should_backtranslate:
            loss_bt_src, loss_bt_trg = self._run_bt(src, trg)
            self.val_losses['loss_bt_src'].append(loss_bt_src.data[0])
            self.val_losses['loss_bt_trg'].append(loss_bt_trg.data[0])
            self.val_iters['loss_bt_src'].append(self.num_iters_done)
            self.val_iters['loss_bt_trg'].append(self.num_iters_done)

        # Discriminator step
        losses, domains_predictions = self._run_discriminator(*encodings)
        discr_loss_src, discr_loss_trg = losses
        self.val_losses['discr_loss_src'].append(discr_loss_src.data[0])
        self.val_losses['discr_loss_trg'].append(discr_loss_trg.data[0])
        self.val_iters['discr_loss_src'].append(self.num_iters_done)
        self.val_iters['discr_loss_trg'].append(self.num_iters_done)

        # Generator step
        gen_loss_src, gen_loss_trg = self._run_generator(*domains_predictions)
        self.val_losses['gen_loss_src'].append(gen_loss_src.data[0])
        self.val_losses['gen_loss_trg'].append(gen_loss_trg.data[0])
        self.val_iters['gen_loss_src'].append(self.num_iters_done)
        self.val_iters['gen_loss_trg'].append(self.num_iters_done)

    def validate(self, val_data):
        for val_batch in val_data:
            self.eval_on_batch(val_batch)
            
    def validate_translation(self, val_data):
        val_losses_src_to_trg = []
        val_losses_trg_to_src = []

        for val_batch in val_data:
            val_src, val_trg = val_batch

            val_pred_trg = self.transformer(val_src, val_trg)
            val_pred_src = self.transformer(val_trg, val_src, use_trg_embs_in_encoder=True, use_src_embs_in_decoder=True)

            val_loss_src_to_trg = self.reconstruct_trg_criterion(val_pred_trg, val_trg[:, 1:].contiguous().view(-1))
            val_loss_trg_to_src = self.reconstruct_src_criterion(val_pred_src, val_src[:, 1:].contiguous().view(-1))

            val_losses_src_to_trg.append(val_loss_src_to_trg.data[0])
            val_losses_trg_to_src.append(val_loss_trg_to_src.data[0])

        self.val_losses['src_to_trg_translation'].append(np.mean(val_losses_src_to_trg))
        self.val_losses['trg_to_src_translation'].append(np.mean(val_losses_trg_to_src))
        self.val_iters['src_to_trg_translation'].append(self.num_iters_done)
        self.val_iters['trg_to_src_translation'].append(self.num_iters_done)
        
        # TODO(universome): we have to add dummy values so plot is displayed
        self.losses['src_to_trg_translation'].append(0)
        self.losses['trg_to_src_translation'].append(0)
            

    def _run_dae(self, src_noised, trg_noised, src, trg):
        # Computing translation for ~src->src and ~trg->trg autoencoding tasks
        preds_src, encodings_src = self.transformer(src_noised, src, return_encodings=True, use_src_embs_in_decoder=True)
        preds_trg, encodings_trg = self.transformer(trg_noised, trg, return_encodings=True, use_trg_embs_in_encoder=True)

        # Computing losses
        dae_loss_src = self.reconstruct_src_criterion(preds_src, src[:, 1:].contiguous().view(-1))
        dae_loss_trg = self.reconstruct_trg_criterion(preds_trg, trg[:, 1:].contiguous().view(-1))

        return (dae_loss_src, dae_loss_trg), (encodings_src, encodings_trg)

    def _run_bt(self, src, trg):
        self.transformer.eval()
        # Get translations for backtranslation
        bt_trg = self.transformer.translate_batch(src, beam_size=2, max_len=self.max_seq_len-2)
        bt_src = self.transformer.translate_batch(trg, beam_size=2, max_len=self.max_seq_len-2,
                                                  use_trg_embs_in_encoder=True, use_src_embs_in_decoder=True)

        bt_trg = [[constants.BOS] + seq for seq in bt_trg]
        bt_src = [[constants.BOS] + seq for seq in bt_src]
        bt_trg = pad_to_longest(bt_trg)
        bt_src = pad_to_longest(bt_src)

        # Computing predictions for back-translated sentences
        self.transformer.train()
        bt_src_preds = self.transformer(bt_trg, src, use_trg_embs_in_encoder=True, use_src_embs_in_decoder=True)
        bt_trg_preds = self.transformer(bt_src, trg)

        # Computing losses
        loss_bt_src = self.reconstruct_src_criterion(bt_src_preds, src[:, 1:].contiguous().view(-1))
        loss_bt_trg = self.reconstruct_trg_criterion(bt_trg_preds, trg[:, 1:].contiguous().view(-1))

        return loss_bt_src, loss_bt_trg

    def _run_discriminator(self, encodings_src, encodings_trg):
        domains_preds_src = self.discriminator(encodings_src.view(-1, 512)).view(-1)
        domains_preds_trg = self.discriminator(encodings_trg.view(-1, 512)).view(-1)

        # Generating targets for discriminator
        true_domains_src = Variable(torch.Tensor([0] * len(domains_preds_src)))
        true_domains_trg = Variable(torch.Tensor([1] * len(domains_preds_trg)))

        if use_cuda:
            true_domains_src = true_domains_src.cuda()
            true_domains_trg = true_domains_trg.cuda()

        # True domains for discriminator loss
        discr_loss_src = self.adv_criterion(domains_preds_src, true_domains_src)
        discr_loss_trg = self.adv_criterion(domains_preds_trg, true_domains_trg)

        return (discr_loss_src, discr_loss_trg), (domains_preds_src, domains_preds_trg)

    def _run_generator(self, domains_preds_src, domains_preds_trg):
        fake_domains_src = Variable(torch.Tensor([1] * len(domains_preds_src)))
        fake_domains_trg = Variable(torch.Tensor([0] * len(domains_preds_trg)))

        if use_cuda:
            fake_domains_src = fake_domains_src.cuda()
            fake_domains_trg = fake_domains_trg.cuda()

        # Faking domains for generator loss
        gen_loss_src = self.adv_criterion(domains_preds_src, fake_domains_src)
        gen_loss_trg = self.adv_criterion(domains_preds_trg, fake_domains_trg)

        return gen_loss_src, gen_loss_trg

    def plot_losses(self):
        clear_output(True)

        losses_pairs = [
            ('dae_loss_src', 'dae_loss_trg'),
            ('loss_bt_src', 'loss_bt_trg'),
            ('discr_loss_src', 'discr_loss_trg'),
            ('gen_loss_src', 'gen_loss_trg'),
            ('src_to_trg_translation', 'trg_to_src_translation')
        ]

        for src, trg in losses_pairs:
            if len(self.losses[src]) == 0 or len(self.losses[trg]) == 0:
                continue

            plt.figure(figsize=[16,4])

            plt.subplot(121)
            plt.title(LOSSES_TITLES[src])
            plt.plot(self.losses[src])
            plt.plot(pd.DataFrame(self.losses[src]).ewm(span=100).mean())
            plt.plot(self.val_iters[src], self.val_losses[src])
            plt.grid()

            plt.subplot(122)
            plt.title(LOSSES_TITLES[trg])
            plt.plot(self.losses[trg])
            plt.plot(pd.DataFrame(self.losses[trg]).ewm(span=100).mean())
            plt.plot(self.val_iters[trg], self.val_losses[trg])
            plt.grid()

        plt.show()
