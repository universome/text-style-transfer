from copy import deepcopy

import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.data_utils import pad_to_longest, token_ids_to_sents
from src.utils.bleu import compute_bleu_for_sents
import src.transformer.constants as constants

use_cuda = torch.cuda.is_available()

SCORES_TITLES = {
    'dae_loss_src': '[src] lang AE loss',
    'dae_loss_trg': '[trg] lang AE loss',
    'loss_bt_src': '[src] lang back-translation loss',
    'loss_bt_trg': '[trg] lang back-translation loss',
    'discr_loss_src': '[src] lang discriminator loss',
    'discr_loss_trg': '[trg] lang discriminator loss',
    'gen_loss_src': '[src] lang generator loss',
    'gen_loss_trg': '[trg] lang generator loss',
    'src_to_trg_translation': '[src->trg] translation loss',
    'trg_to_src_translation': '[trg->src] translation loss',
    'src_to_trg_bleu': '[src->trg] BLEU score',
    'trg_to_src_bleu': '[trg->src] BLEU score',
}


class Trainer:
    def __init__(self, transformer, discriminator, vocab_src, vocab_trg,
                 transformer_optimizer, discriminator_optimizer,
                 reconstruct_src_criterion, reconstruct_trg_criterion, adv_criterion, config):

        self.transformer = transformer
        self.discriminator = discriminator
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.transformer_optimizer = transformer_optimizer
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

        self.alternate_adv_training = config.get('alternate_adv_training', False)
        self.alternate_adv_training_iters = config.get('alternate_adv_training_iters', 100)

        self.train_scores = {
            'dae_loss_src': [],
            'dae_loss_trg': [],
            'loss_bt_src': [],
            'loss_bt_trg': [],
            'discr_loss_src': [],
            'discr_loss_trg': [],
            'gen_loss_src': [],
            'gen_loss_trg': [],
            'src_to_trg_bleu': [],
            'trg_to_src_bleu': []
        }

        # Val losses have the same structure, so let's just clone them
        self.val_scores = deepcopy(self.train_scores)
        self.val_iters = deepcopy(self.train_scores)

        # Additionally, we have val scores for translation CE
        self.val_scores['src_to_trg_translation'] = []
        self.val_scores['trg_to_src_translation'] = []
        self.val_iters['src_to_trg_translation'] = []
        self.val_iters['trg_to_src_translation'] = []

    def run_training(self, training_data, val_data, translation_val_data,
                     plot_every=50, val_translate_every=100):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            self.validate(val_data)

            for batch in tqdm(training_data, leave=False):
                try:
                    self.train_on_batch(batch)
                    if self.num_iters_done % val_translate_every == 0: self.validate_translation(translation_val_data)
                    if self.num_iters_done % plot_every == 0: self.plot_scores()
                    self.num_iters_done += 1
                except KeyboardInterrupt:
                    should_continue = False
                    break

            self.num_epochs_done += 1

    def train_on_batch(self, batch):
        src_noised, trg_noised, src, trg = batch
        should_backtranslate = self.num_iters_done >= self.start_bt_from_iter
        should_train_generator = (self.num_iters_done // self.alternate_adv_training_iters) % 2 == 1
        total_transformer_loss = 0
        total_discriminator_loss = 0

        # DAE step
        self.transformer.train()
        losses, encodings = self._run_dae(src_noised, trg_noised, src, trg)
        dae_loss_src, dae_loss_trg = losses
        total_transformer_loss += (dae_loss_src + dae_loss_trg)
        self.train_scores['dae_loss_src'].append(dae_loss_src.data[0])
        self.train_scores['dae_loss_trg'].append(dae_loss_trg.data[0])

        # (Back) translation step
        if should_backtranslate:
            loss_bt_src, loss_bt_trg = self._run_bt(src, trg)
            total_transformer_loss += (loss_bt_src + loss_bt_trg)
            self.train_scores['loss_bt_src'].append(loss_bt_src.data[0])
            self.train_scores['loss_bt_trg'].append(loss_bt_trg.data[0])

        # Discriminator step
        losses, domains_predictions = self._run_discriminator(*encodings)
        discr_loss_src, discr_loss_trg = losses
        total_discriminator_loss += (discr_loss_src + discr_loss_trg)
        self.train_scores['discr_loss_src'].append(discr_loss_src.data[0])
        self.train_scores['discr_loss_trg'].append(discr_loss_trg.data[0])

        # Generator step
        gen_loss_src, gen_loss_trg = self._run_generator(*domains_predictions)
        if self.alternate_adv_training and should_train_generator:
            total_transformer_loss += (gen_loss_src + gen_loss_trg)
        self.train_scores['gen_loss_src'].append(gen_loss_src.data[0])
        self.train_scores['gen_loss_trg'].append(gen_loss_trg.data[0])

        # Backward passes
        self.transformer_optimizer.zero_grad()
        total_transformer_loss.backward(retain_graph=True)
        self.transformer_optimizer.step()

        if self.alternate_adv_training and not should_train_generator:
            self.discriminator_optimizer.zero_grad()
            total_discriminator_loss.backward()
            self.discriminator_optimizer.step()

    def eval_on_batch(self, batch):
        src_noised, trg_noised, src, trg = batch
        should_backtranslate = self.num_iters_done >= self.start_bt_from_iter

        self.transformer.eval()
        self.discriminator.eval()

        # DAE step
        losses, encodings = self._run_dae(src_noised, trg_noised, src, trg)
        dae_loss_src, dae_loss_trg = losses
        self.val_scores['dae_loss_src'].append(dae_loss_src.data[0])
        self.val_scores['dae_loss_trg'].append(dae_loss_trg.data[0])
        self.val_iters['dae_loss_src'].append(self.num_iters_done)
        self.val_iters['dae_loss_trg'].append(self.num_iters_done)

        # (Back) translation step
        if should_backtranslate:
            loss_bt_src, loss_bt_trg = self._run_bt(src, trg)
            self.val_scores['loss_bt_src'].append(loss_bt_src.data[0])
            self.val_scores['loss_bt_trg'].append(loss_bt_trg.data[0])
            self.val_iters['loss_bt_src'].append(self.num_iters_done)
            self.val_iters['loss_bt_trg'].append(self.num_iters_done)

        # Discriminator step
        losses, domains_predictions = self._run_discriminator(*encodings)
        discr_loss_src, discr_loss_trg = losses
        self.val_scores['discr_loss_src'].append(discr_loss_src.data[0])
        self.val_scores['discr_loss_trg'].append(discr_loss_trg.data[0])
        self.val_iters['discr_loss_src'].append(self.num_iters_done)
        self.val_iters['discr_loss_trg'].append(self.num_iters_done)

        # Generator step
        gen_loss_src, gen_loss_trg = self._run_generator(*domains_predictions)
        self.val_scores['gen_loss_src'].append(gen_loss_src.data[0])
        self.val_scores['gen_loss_trg'].append(gen_loss_trg.data[0])
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

        self.val_scores['src_to_trg_translation'].append(np.mean(val_losses_src_to_trg))
        self.val_scores['trg_to_src_translation'].append(np.mean(val_losses_trg_to_src))
        self.val_iters['src_to_trg_translation'].append(self.num_iters_done)
        self.val_iters['trg_to_src_translation'].append(self.num_iters_done)

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

        # We should prepend our sentences with BOS symbol
        bt_trg = [[constants.BOS] + s for s in bt_trg]
        bt_src = [[constants.BOS] + s for s in bt_src]

        # It's a good opportunity for us to measure BLEU score
        bt_trg_sents = token_ids_to_sents(bt_trg, self.vocab_trg)
        bt_src_sents = token_ids_to_sents(bt_src, self.vocab_src)
        src_sents = token_ids_to_sents(src, self.vocab_src)
        trg_sents = token_ids_to_sents(trg, self.vocab_trg)

        self.train_scores['src_to_trg_bleu'].append(compute_bleu_for_sents(bt_trg_sents, trg_sents))
        self.train_scores['trg_to_src_bleu'].append(compute_bleu_for_sents(bt_src_sents, src_sents))

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

    def plot_scores(self):
        clear_output(True)

        losses_pairs = [
            ('dae_loss_src', 'dae_loss_trg'),
            ('loss_bt_src', 'loss_bt_trg'),
            ('discr_loss_src', 'discr_loss_trg'),
            ('gen_loss_src', 'gen_loss_trg')
        ]

        for src, trg in losses_pairs:
            if len(self.train_scores[src]) == 0 or len(self.train_scores[trg]) == 0:
                continue

            plt.figure(figsize=[16,4])

            plt.subplot(121)
            plt.title(SCORES_TITLES[src])
            plt.plot(self.train_scores[src])
            plt.plot(pd.DataFrame(self.train_scores[src]).ewm(span=100).mean())
            plt.plot(self.val_iters[src], self.val_scores[src])
            plt.grid()

            plt.subplot(122)
            plt.title(SCORES_TITLES[trg])
            plt.plot(self.train_scores[trg])
            plt.plot(pd.DataFrame(self.train_scores[trg]).ewm(span=100).mean())
            plt.plot(self.val_iters[trg], self.val_scores[trg])
            plt.grid()

        # We have two additional plots: train BLEU and validation CE
        self.make_plots_for_bleu_and_translation_val()

        plt.show()

    # TODO: just add display style parameter: single/paired
    def make_plots_for_bleu_and_translation_val(self):
        if not self.val_scores['src_to_trg_translation'] and not self.train_scores['src_to_trg_bleu']:
            # We have nothing to display :(
            return

        plt.figure(figsize=[16,4])

        # TODO: show it for validation too!
        if self.train_scores['src_to_trg_bleu']:
            src, trg = 'src_to_trg_bleu', 'trg_to_src_bleu'
            iters = np.arange(self.start_bt_from_iter, self.start_bt_from_iter + len(self.train_scores[src]))
            plt.subplot(121)
            plt.title('Train BLEU score')
            plt.plot(iters, pd.DataFrame(self.train_scores[src]).ewm(span=100).mean(), label=SCORES_TITLES[src])
            plt.plot(iters, pd.DataFrame(self.train_scores[trg]).ewm(span=100).mean(), label=SCORES_TITLES[trg])
            plt.grid()
            plt.legend()

        if self.val_scores['src_to_trg_translation']:
            src, trg = 'src_to_trg_translation', 'trg_to_src_translation'
            plt.subplot(122)
            plt.title('Val translation loss')
            plt.plot(self.val_iters[src], self.val_scores[src], label=SCORES_TITLES[src])
            plt.plot(self.val_iters[trg], self.val_scores[trg], label=SCORES_TITLES[trg])
            plt.grid()
            plt.legend()
