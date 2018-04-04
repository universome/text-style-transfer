from copy import deepcopy

import torch
from torch.autograd import Variable
from tqdm import tqdm; tqdm.monitor_interval = 0
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.data_utils import pad_to_longest, token_ids_to_sents
from src.utils.bleu import compute_bleu_for_sents
from src.vocab import constants
from src.utils.common import variable

use_cuda = torch.cuda.is_available()

SCORES_TITLES = {
    'dae_loss_src': '[src] lang AE loss',
    'dae_loss_trg': '[trg] lang AE loss',
    'loss_bt_src': '[src] lang back-translation loss',
    'loss_bt_trg': '[trg] lang back-translation loss',
    'src_to_trg_bleu': '[src->trg] BLEU score',
    'trg_to_src_bleu': '[trg->src] BLEU score',
}


class UMTTrainer:
    def __init__(self, transformer, transformer_optimizer, vocab_src, vocab_trg,
                 reconstruct_src_criterion, reconstruct_trg_criterion, config):

        self.transformer = transformer
        self.transformer_optimizer = transformer_optimizer
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.reconstruct_src_criterion = reconstruct_src_criterion
        self.reconstruct_trg_criterion = reconstruct_trg_criterion

        if use_cuda:
            self.transformer.cuda()
            self.reconstruct_src_criterion.cuda()
            self.reconstruct_trg_criterion.cuda()

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 100)
        self.start_bt_from_iter = config.get('start_bt_from_iter', 500)
        self.max_seq_len = config.get('max_seq_len', 50)

        self.train_scores = {
            'dae_loss_src': [],
            'dae_loss_trg': [],
            'loss_bt_src': [],
            'loss_bt_trg': [],
        }

        self.val_scores = {'src_to_trg_bleu': [], 'trg_to_src_bleu': []}

    def run_training(self, training_data, val_data,
                     plot_every=50, val_bleu_every=500):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            try:
                for batch in tqdm(training_data):
                    self.train_on_batch(batch)
                    if self.num_iters_done % val_bleu_every == 0: self.validate_bleu(val_data)
                    if self.num_iters_done % plot_every == 0: self.plot_scores()
                    self.num_iters_done += 1

            except KeyboardInterrupt:
                should_continue = False
                break

            self.num_epochs_done += 1

    def train_on_batch(self, batch):
        src_noised, trg_noised, src, trg = batch
        should_backtranslate = self.num_iters_done >= self.start_bt_from_iter
        total_transformer_loss = 0
        total_discriminator_loss = 0

        # DAE step
        self.transformer.train()
        dae_loss_src, dae_loss_trg = self._run_dae(src_noised, trg_noised, src, trg)
        total_transformer_loss += (dae_loss_src + dae_loss_trg)

        # (Back) translation step
        if should_backtranslate:
            loss_bt_src, loss_bt_trg = self._run_bt(src, trg)
            total_transformer_loss += (loss_bt_src + loss_bt_trg)
            self.train_scores['loss_bt_src'].append(loss_bt_src.data[0])
            self.train_scores['loss_bt_trg'].append(loss_bt_trg.data[0])

        # Discriminator step
        # We should pick elements which are not PADs
        #domains_preds_src = self.discriminator(encodings_src, src_noised).squeeze()
        #domains_preds_trg = self.discriminator(encodings_trg, trg_noised).squeeze()

        # Training discriminator on true domains
        #discr_loss_src = self.adv_criterion(domains_preds_src, torch.zeros_like(domains_preds_src))
        #discr_loss_trg = self.adv_criterion(domains_preds_trg, torch.ones_like(domains_preds_trg))
        #total_discriminator_loss += (discr_loss_src + discr_loss_trg)

        # Training generator on fake domains
        #gen_loss_src = self.adv_criterion(domains_preds_src, torch.ones_like(domains_preds_src))
        #gen_loss_trg = self.adv_criterion(domains_preds_trg, torch.zeros_like(domains_preds_trg))
        #total_transformer_loss += ((gen_loss_src + gen_loss_trg) * self.gen_loss_coef)

        # Backward passes
        self.transformer_optimizer.zero_grad()
        total_transformer_loss.backward(retain_graph=True)
        self.transformer_optimizer.step()

        #self.discriminator_optimizer.zero_grad()
        #total_discriminator_loss.backward()
        #self.discriminator_optimizer.step()

        # Saving metrics
        self.train_scores['dae_loss_src'].append(dae_loss_src.data[0])
        self.train_scores['dae_loss_trg'].append(dae_loss_trg.data[0])
        #self.train_scores['discr_loss_src'].append(discr_loss_src.data[0])
        #self.train_scores['discr_loss_trg'].append(discr_loss_trg.data[0])
        #self.train_scores['gen_loss_src'].append(gen_loss_src.data[0])
        #self.train_scores['gen_loss_trg'].append(gen_loss_trg.data[0])

    def validate_bleu(self, val_data, return_results=False):
        all_translations_src_to_trg = []
        all_translations_trg_to_src = []
        all_targets_src_to_trg = []
        all_targets_trg_to_src = []

        for batch in val_data:
            translations_src_to_trg = self.transformer.translate_batch(batch[0], max_len=self.max_seq_len, beam_size=4)
            translations_trg_to_src = self.transformer.translate_batch(batch[1], max_len=self.max_seq_len, beam_size=4,
                                                                       use_src_embs_in_decoder=True,
                                                                       use_trg_embs_in_encoder=True)

            all_translations_src_to_trg += token_ids_to_sents(translations_src_to_trg, self.vocab_trg)
            all_translations_trg_to_src += token_ids_to_sents(translations_trg_to_src, self.vocab_src)
            all_targets_src_to_trg += token_ids_to_sents(batch[1], self.vocab_trg)
            all_targets_trg_to_src += token_ids_to_sents(batch[0], self.vocab_src)

        bleu_src_to_trg = compute_bleu_for_sents(all_translations_src_to_trg, all_targets_src_to_trg)
        bleu_trg_to_src = compute_bleu_for_sents(all_translations_trg_to_src, all_targets_trg_to_src)

        if return_results:
            scores = (bleu_src_to_trg, bleu_trg_to_src)
            translations = {
                'translations_src_to_trg': all_translations_src_to_trg,
                'translations_trg_to_src': all_translations_trg_to_src,
                'targets_src_to_trg': all_targets_src_to_trg,
                'targets_trg_to_src': all_targets_trg_to_src
            }

            return scores, translations
        else:
            self.val_scores['src_to_trg_bleu'].append(bleu_src_to_trg)
            self.val_scores['trg_to_src_bleu'].append(bleu_trg_to_src)

    def _run_dae(self, src_noised, trg_noised, src, trg):
        # Computing translation for ~src->src and ~trg->trg autoencoding tasks
        preds_src = self.transformer(src_noised, src, 'src')
        preds_trg = self.transformer(trg_noised, trg, 'trg')

        # Computing losses
        dae_loss_src = self.reconstruct_src_criterion(preds_src, src[:, 1:].contiguous().view(-1))
        dae_loss_trg = self.reconstruct_trg_criterion(preds_trg, trg[:, 1:].contiguous().view(-1))

        return dae_loss_src, dae_loss_trg

    def _run_bt(self, src, trg):
        self.transformer.eval()
        # Get translations for backtranslation
        bt_src_to_trg = self.transformer.translate_batch(src, 'trg', beam_size=1, max_len=self.max_seq_len-2)
        bt_trg_to_src = self.transformer.translate_batch(trg, 'src', beam_size=1, max_len=self.max_seq_len-2)

        # We should prepend our sentences with BOS symbol
        bt_src_to_trg = pad_to_longest([[constants.BOS] + s for s in bt_src_to_trg])
        bt_trg_to_src = pad_to_longest([[constants.BOS] + s for s in bt_trg_to_src])

        # Computing predictions for back-translated sentences
        self.transformer.train()
        bt_src_to_trg_to_src_preds = self.transformer(bt_src_to_trg, src, 'src')
        bt_trg_to_src_to_trg_preds = self.transformer(bt_trg_to_src, trg, 'trg')

        # Computing losses
        loss_bt_src = self.reconstruct_src_criterion(bt_src_to_trg_to_src_preds, src[:, 1:].contiguous().view(-1))
        loss_bt_trg = self.reconstruct_trg_criterion(bt_trg_to_src_to_trg_preds, trg[:, 1:].contiguous().view(-1))

        return loss_bt_src, loss_bt_trg

    def plot_scores(self):
        clear_output(True)

        losses_to_display = [
            ('dae_loss_src', 'dae_loss_trg', 221),
            ('loss_bt_src', 'loss_bt_trg', 222),
            #('discr_loss_src', 'discr_loss_trg', 223),
            #('gen_loss_src', 'gen_loss_trg', 224)
        ]

        plt.figure(figsize=[16,8])

        for src, trg, plot_position in losses_to_display:
            if len(self.train_scores[src]) == 0 or len(self.train_scores[trg]) == 0:
                continue

            plt.subplot(plot_position)

            plt.plot(self.train_scores[src], color='#33ACFF')
            plt.plot(self.train_scores[trg], color='#FF7B7B')

            plt.plot(pd.DataFrame(self.train_scores[src]).ewm(span=50).mean(), label=SCORES_TITLES[src], color='#0000FF')
            plt.plot(pd.DataFrame(self.train_scores[trg]).ewm(span=50).mean(), label=SCORES_TITLES[trg], color='#FF0000')

            plt.legend(loc='lower left')
            plt.grid()

        plt.show()

        self.plot_validation_scores()

    def plot_validation_scores(self):
        if not self.val_scores['src_to_trg_bleu']: return

        plt.figure(figsize=[16,4])

        src, trg = 'src_to_trg_bleu', 'trg_to_src_bleu'
        val_iters = np.arange(len(self.val_scores[src])) * (self.num_iters_done / len(self.val_scores[src]))

        plt.title('Val translation BLEU')
        plt.plot(val_iters, self.val_scores[src], label=SCORES_TITLES[src])
        plt.plot(val_iters, self.val_scores[trg], label=SCORES_TITLES[trg])
        plt.grid()
        plt.legend()

        plt.show()
