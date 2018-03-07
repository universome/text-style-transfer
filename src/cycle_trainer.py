from copy import deepcopy

import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.data_utils import token_ids_to_sents
from src.utils.bleu import compute_bleu_for_sents


use_cuda = torch.cuda.is_available()

SCORES_TITLES = {
    # Training metrics
    'src_reconstruction_loss': '[src->trg->src] cycle loss',
    'trg_reconstruction_loss': '[trg->src->trg] cycle loss',
    'gen_src_to_trg_loss': '[src->trg] generator loss',
    'gen_trg_to_src_loss': '[trg->src] generator loss',
    'discr_src_loss_on_true': '[src] discriminator loss on true data',
    'discr_src_loss_on_fake': '[src] discriminator loss on fake data',
    'discr_trg_loss_on_true': '[trg] discriminator loss on true data',
    'discr_trg_loss_on_fake': '[trg] discriminator loss on fake data',

    # Validation metrics
    'src_to_trg_bleu': '[src->trg] BLEU score',
    'trg_to_src_bleu': '[trg->src] BLEU score'
}


class CycleTrainer:
    def __init__(self, transformer_src_to_trg, transformer_trg_to_src,
                 discriminator_src, discriminator_trg, vocab_src, vocab_trg,
                 transformer_src_to_trg_optimizer, transformer_trg_to_src_optimizer,
                 discriminator_src_optimizer, discriminator_trg_optimizer,
                 reconstruct_src_criterion, reconstruct_trg_criterion, adv_criterion, config):

        self.transformer_src_to_trg = transformer_src_to_trg
        self.transformer_trg_to_src = transformer_trg_to_src
        self.discriminator_src = discriminator_src
        self.discriminator_trg = discriminator_trg
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.transformer_src_to_trg_optimizer = transformer_src_to_trg_optimizer
        self.transformer_trg_to_src_optimizer = transformer_trg_to_src_optimizer
        self.discriminator_src_optimizer = discriminator_src_optimizer
        self.discriminator_trg_optimizer = discriminator_trg_optimizer
        self.reconstruct_src_criterion = reconstruct_src_criterion
        self.reconstruct_trg_criterion = reconstruct_trg_criterion
        self.adv_criterion = adv_criterion

        if use_cuda:
            self.transformer_src_to_trg.cuda()
            self.transformer_trg_to_src.cuda()
            self.discriminator_src.cuda()
            self.discriminator_trg.cuda()
            self.reconstruct_src_criterion.cuda()
            self.reconstruct_trg_criterion.cuda()
            self.adv_criterion.cuda()

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 100)
        self.max_len = config.get('max_len', 50)
        self.temperature_update_scheme = config.get('temperature_update_scheme', (1,1,0))
        self.generator_loss_coef_update_scheme = config.get('generator_loss_coef_update_scheme', (1,1,0))

        self.train_scores = {
            'discr_src_loss_on_true': [],
            'discr_src_loss_on_fake': [],
            'discr_trg_loss_on_true': [],
            'discr_trg_loss_on_fake': [],
            'gen_src_to_trg_loss': [],
            'gen_trg_to_src_loss': [],
            'src_reconstruction_loss': [],
            'trg_reconstruction_loss': []
        }

        self.val_scores = {
            'src_to_trg_bleu': [],
            'trg_to_src_bleu': []
        }
        self.val_iters = []

    def run_training(self, training_data, val_data, plot_every=50, val_bleu_every=100):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            try:
                for batch in tqdm(training_data, leave=False):
                    self.train_on_batch(batch)
                    if self.num_iters_done % val_bleu_every == 0: self.validate_bleu(val_data)
                    if self.num_iters_done % plot_every == 0: self.plot_scores()
                    self.num_iters_done += 1
            except KeyboardInterrupt:
                should_continue = False
                break

            self.num_epochs_done += 1

    def train_on_batch(self, batch):
        src, trg = batch
        # Normal forward pass
        preds_src_to_trg = self.transformer_src_to_trg.differentiable_translate(src, self.vocab_trg, max_len=30, temperature=self.temperature())
        preds_trg_to_src = self.transformer_trg_to_src.differentiable_translate(trg, self.vocab_src, max_len=30, temperature=self.temperature())

        # Running our discriminators to predict domains
        # Target discriminator
        true_domains_preds_trg = self.discriminator_trg(trg)
        fake_domains_preds_trg = self.discriminator_trg(preds_src_to_trg, one_hot_input=True)
        true_domains_preds_src = self.discriminator_src(src)
        fake_domains_preds_src = self.discriminator_src(preds_trg_to_src, one_hot_input=True)

        true_domains_y_trg = Variable(torch.zeros(len(trg)))
        fake_domains_y_trg = Variable(torch.ones(len(preds_src_to_trg)))
        true_domains_y_src = Variable(torch.zeros(len(src)))
        fake_domains_y_src = Variable(torch.ones(len(preds_trg_to_src)))

        # Revert classes for generator
        fake_domains_y_trg_for_gen = Variable(torch.zeros(len(preds_src_to_trg)))
        fake_domains_y_src_for_gen = Variable(torch.zeros(len(preds_trg_to_src)))

        if use_cuda:
            true_domains_y_trg = true_domains_y_trg.cuda()
            fake_domains_y_trg = fake_domains_y_trg.cuda()
            true_domains_y_src = true_domains_y_src.cuda()
            fake_domains_y_src = fake_domains_y_src.cuda()
            fake_domains_y_trg_for_gen = fake_domains_y_trg_for_gen.cuda()
            fake_domains_y_src_for_gen = fake_domains_y_src_for_gen.cuda()

        discr_src_loss_on_true = self.adv_criterion(true_domains_preds_src, true_domains_y_src)
        discr_src_loss_on_fake = self.adv_criterion(fake_domains_preds_src, fake_domains_y_src)
        discr_trg_loss_on_true = self.adv_criterion(true_domains_preds_trg, true_domains_y_trg)
        discr_trg_loss_on_fake = self.adv_criterion(fake_domains_preds_trg, fake_domains_y_trg)
        discr_src_loss = discr_src_loss_on_true + discr_src_loss_on_fake
        discr_trg_loss = discr_trg_loss_on_true + discr_trg_loss_on_fake

        # Uff, ok. Let's compute losses for our generators
        gen_src_to_trg_loss = self.adv_criterion(fake_domains_preds_trg, fake_domains_y_trg_for_gen) * self.generator_loss_coef()
        gen_trg_to_src_loss = self.adv_criterion(fake_domains_preds_src, fake_domains_y_src_for_gen) * self.generator_loss_coef()

        # "Back-translation" passes
        preds_src_to_trg_to_src = self.transformer_trg_to_src(preds_src_to_trg, src, one_hot_src=True)
        preds_trg_to_src_to_trg = self.transformer_src_to_trg(preds_trg_to_src, trg, one_hot_src=True)

        # Trying to reconstruct what we have just back-translated
        src_reconstruction_loss = self.reconstruct_src_criterion(preds_src_to_trg_to_src, src[:, 1:].contiguous().view(-1))
        trg_reconstruction_loss = self.reconstruct_trg_criterion(preds_trg_to_src_to_trg, trg[:, 1:].contiguous().view(-1))

        ### Update weights ###
        # Let's update discriminators first and forget about them instantly
        self.discriminator_src_optimizer.zero_grad()
        self.discriminator_trg_optimizer.zero_grad()

        discr_src_loss.backward(retain_graph=True)
        discr_trg_loss.backward(retain_graph=True)

        self.discriminator_src_optimizer.step()
        self.discriminator_trg_optimizer.step()

        # Now let's optimize reconstruction and generator losses
        self.transformer_src_to_trg_optimizer.zero_grad()
        self.transformer_trg_to_src_optimizer.zero_grad()

        src_reconstruction_loss.backward(retain_graph=True)
        trg_reconstruction_loss.backward(retain_graph=True)
        gen_src_to_trg_loss.backward(retain_graph=True)
        gen_trg_to_src_loss.backward()

        self.transformer_src_to_trg_optimizer.step()
        self.transformer_trg_to_src_optimizer.step()

        # Saving metrics
        self.train_scores['discr_src_loss_on_true'].append(discr_src_loss_on_true.data[0])
        self.train_scores['discr_src_loss_on_fake'].append(discr_src_loss_on_fake.data[0])
        self.train_scores['discr_trg_loss_on_true'].append(discr_trg_loss_on_true.data[0])
        self.train_scores['discr_trg_loss_on_fake'].append(discr_trg_loss_on_fake.data[0])
        self.train_scores['gen_src_to_trg_loss'].append(gen_src_to_trg_loss.data[0] / self.generator_loss_coef())
        self.train_scores['gen_trg_to_src_loss'].append(gen_trg_to_src_loss.data[0] / self.generator_loss_coef())
        self.train_scores['src_reconstruction_loss'].append(src_reconstruction_loss.data[0])
        self.train_scores['trg_reconstruction_loss'].append(trg_reconstruction_loss.data[0])

    def validate_bleu(self, val_data, max_len=50, beam_size=1, return_translations=False):
        all_translations_src_to_trg = []
        all_translations_trg_to_src = []
        all_targets_src_to_trg = []
        all_targets_trg_to_src = []

        for batch in val_data:
            translations_src_to_trg = self.transformer_src_to_trg.translate_batch(batch[0], max_len=max_len, beam_size=beam_size)
            translations_trg_to_src = self.transformer_trg_to_src.translate_batch(batch[1], max_len=max_len, beam_size=beam_size)

            all_translations_src_to_trg += token_ids_to_sents(translations_src_to_trg, self.vocab_trg)
            all_translations_trg_to_src += token_ids_to_sents(translations_trg_to_src, self.vocab_src)
            all_targets_src_to_trg += token_ids_to_sents(batch[1], self.vocab_trg)
            all_targets_trg_to_src += token_ids_to_sents(batch[0], self.vocab_src)

        src_to_trg_bleu = compute_bleu_for_sents(all_translations_src_to_trg, all_targets_src_to_trg)
        trg_to_src_bleu = compute_bleu_for_sents(all_translations_trg_to_src, all_targets_trg_to_src)

        self.val_scores['src_to_trg_bleu'].append(src_to_trg_bleu)
        self.val_scores['trg_to_src_bleu'].append(trg_to_src_bleu)
        self.val_iters.append(self.num_iters_done)
        
        if return_translations:
            return {
                'all_translations_src_to_trg': all_translations_src_to_trg,
                'all_translations_trg_to_src': all_translations_trg_to_src,
                'all_targets_src_to_trg': all_targets_src_to_trg,
                'all_targets_trg_to_src': all_targets_trg_to_src
            }

    def plot_scores(self):
        clear_output(True)

        losses_to_display = [
            ('src_reconstruction_loss', 'trg_reconstruction_loss', 221),
            ('gen_src_to_trg_loss', 'gen_trg_to_src_loss', 222),
            ('discr_src_loss_on_true', 'discr_src_loss_on_fake', 223),
            ('discr_trg_loss_on_true', 'discr_trg_loss_on_fake', 224)
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

        # Let's also plot validation BLEU scores
        plt.figure(figsize=[8,4])
        plt.title('Validation BLEU')
        plt.plot(self.val_iters, np.array(self.val_scores['src_to_trg_bleu']), label='[src->trg] BLEU')
        plt.plot(self.val_iters, np.array(self.val_scores['trg_to_src_bleu']), label='[trg->src] BLEU')
        plt.grid()
        plt.legend()
        plt.show()
        
    def temperature(self):
        return compute_param_by_scheme(self.temperature_update_scheme, self.num_iters_done)
        
    def generator_loss_coef(self):
        return compute_param_by_scheme(self.generator_loss_coef_update_scheme, self.num_iters_done)
        
        
def compute_param_by_scheme(scheme, num_iters_done):
    """
    Arguments:
    - scheme: format (start_val, end_val, period)
    """
    t1, t2, period = scheme
    
    if num_iters_done > period:
        return t2
    else:
        return t1 - (t1 - t2) * num_iters_done / period
    