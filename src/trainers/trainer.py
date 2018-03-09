import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm; tqdm.monitor_interval = 0

from src.utils.common import variable
from src.utils.data_utils import pad_to_longest, token_ids_to_sents
import src.transformer.constants as constants
from src.utils.bleu import compute_bleu_for_sents


# TODO(universome): passing tasks as a list looks like
# not the best architecture design
AVAILABLE_TASKS = [
    # Use back-tranlslated sentences to aid learning
    'backtranslation',

    # Train as denoising auto-encoder
    'dae',

    # Use cycle loss
    # Requires differentiable sampling
    'cycle_loss',

    # Should try to align encodings into the single space
    'discriminate_encodings',

    # Should penalize samples with discriminator loss
    # Requires differentiable sampling
    'discriminate_translations'
]

SCORES_TITLES = {
    # Training metrics
    'dae_src_loss': '[~src->src] DAE loss',
    'dae_trg_loss': '[~trg->trg] DAE loss',
    'gen_src_to_trg_loss': '[src->trg] generator loss',
    'gen_trg_to_src_loss': '[trg->src] generator loss',
    'discr_src_loss_on_true': 'discriminator_src loss on true data',
    'discr_src_loss_on_fake': 'discriminator_src loss on fake data',
    'discr_trg_loss_on_true': 'discriminator_trg loss on true data',
    'discr_trg_loss_on_fake': 'discriminator_trg loss on fake data',

    # Validation metrics
    'bleu_src_to_trg_to_src': '[src->trg->src] BLEU score',
    'bleu_trg_to_src_to_trg': '[trg->src->trg] BLEU score'
}

class Trainer:
    """
    This is a general trainer for NMT/style-transfer,
    which can handle a lot of experiment setups
    """
    def __init__(self, tasks, models, optimizers, criterions, vocabs, config):
        self.should_discriminate_translations = 'discriminate_translations' in tasks
        self.should_dae = 'dae' in tasks

        # TODO(universome): implement separately shared encoder/decoder
        self.is_translator_shared = config.get('is_translator_shared', False)
        self.is_vocab_shared = config.get('is_vocab_shared', False)

        if 'backtranslation' in tasks: raise NotImplemented
        if 'cycle_loss' in tasks: raise NotImplemented
        if 'align_encodings' in tasks: raise NotImplemented

        if self.is_translator_shared:
            self.translator = models['translator']
            self.translator_optimizer = optimizers['translator_optimizer']
        else:
            self.translator_src_to_trg = models['translator_src_to_trg']
            self.translator_trg_to_src = models['translator_trg_to_src']

            self.translator_src_to_trg_optimizer = optimizers['translator_src_to_trg_optimizer']
            self.translator_trg_to_src_optimizer = optimizers['translator_trg_to_src_optimizer']

        if self.is_vocab_shared:
            self.reconstruction_criterion = criterions['reconstruction_criterion']
        else:
            # We use different criterions for different vocabs, because
            # there are different amount of classes,
            # and class weights vectors have different lengths
            self.reconstruct_src_criterion = criterions['reconstruct_src_criterion']
            self.reconstruct_trg_criterion = criterions['reconstruct_trg_criterion']

        if self.should_discriminate_translations:
            self.discriminator_src = models['discriminator_src']
            self.discriminator_trg = models['discriminator_trg']

            self.discriminator_src_optimizer = optimizers['discriminator_src_optimizer']
            self.discriminator_trg_optimizer = optimizers['discriminator_trg_optimizer']

            # TODO(universome): can we have different adversarial
            # criterions for different discriminators?
            self.adv_criterion = criterions['adv_criterion']

            self.generator_loss_coef_update_scheme = config.get('generator_loss_coef_update_scheme', (1,1,0))
            self.temperature_update_scheme = config.get('temperature_update_scheme', (1,1,0))

        self.train_scores = {}

        if self.should_dae:
            self.train_scores['dae_src_loss'] = []
            self.train_scores['dae_trg_loss'] = []

        if self.should_discriminate_translations:
            self.train_scores['gen_src_to_trg_loss'] = []
            self.train_scores['gen_trg_to_src_loss'] = []
            self.train_scores['discr_src_loss_on_true'] = []
            self.train_scores['discr_src_loss_on_fake'] = []
            self.train_scores['discr_trg_loss_on_true'] = []
            self.train_scores['discr_trg_loss_on_fake'] = []

        self.val_scores = {
            'bleu_src_to_trg_to_src': [],
            'bleu_trg_to_src_to_trg': []
        }

        if self.is_vocab_shared:
            self.vocab = vocabs['vocab']
        else:
            self.vocab_src = vocabs['vocab_src']
            self.vocab_trg = vocabs['vocab_trg']

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_len = config.get('max_len', 50)
        self.max_num_epochs = config.get('max_num_epochs', 100)
        self.log_file = config.get('log_file')
        self.should_log_trg_to_src = config.get('should_log_trg_to_src', False)
        self.plot_every = config.get('plot_every')
        self.val_bleu_every = config.get('val_bleu_every')

    def run_training(self, training_data, val_data):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            try:
                for batch in tqdm(training_data, leave=False):
                    self.train_on_batch(batch)

                    if self.val_bleu_every and self.num_iters_done % self.val_bleu_every == 0:
                        self.validate_bleu(val_data)

                    if self.plot_every and self.num_iters_done % self.plot_every == 0:
                        self.plot_scores()

                    self.num_iters_done += 1
            except KeyboardInterrupt:
                should_continue = False
                break

            self.num_epochs_done += 1

    def train_on_batch(self, batch):
        self._train_mode()

        # TODO: ok, I got tired of making multi-setup logic
        # so from now on I hardcode only DAE + discriminate_translations tasks
        # with shared translator and shared vocab. Arrghhh, that's a shame :(
        src_noised, trg_noised, src, trg = batch

        translator_loss = 0

        dae_src_loss, dae_trg_loss = self._run_dae(src_noised, trg_noised, src, trg)
        translator_loss += (dae_src_loss + dae_trg_loss)

        # Translate sentences into new style
        preds_src_to_trg = self.translator.differentiable_translate(
            src, self.vocab, max_len=self.max_len, temperature=self.temperature())
        preds_trg_to_src = self.translator.differentiable_translate(
            trg, self.vocab, max_len=self.max_len, temperature=self.temperature(),
            use_src_embs_in_decoder=False, use_trg_embs_in_encoder=False)

        # Running our discriminators to predict domains
        # Target discriminator
        true_domains_preds_trg = self.discriminator_trg(trg)
        fake_domains_preds_trg = self.discriminator_trg(preds_src_to_trg, one_hot_input=True)
        true_domains_preds_src = self.discriminator_src(src)
        fake_domains_preds_src = self.discriminator_src(preds_trg_to_src, one_hot_input=True)
        
        # TODO: does it really help? we still need a backward pass for these vars
        del preds_src_to_trg
        del preds_trg_to_src

        true_domains_y_trg = variable(torch.zeros(len(trg)))
        fake_domains_y_trg = variable(torch.ones(len(src)))
        true_domains_y_src = variable(torch.zeros(len(src)))
        fake_domains_y_src = variable(torch.ones(len(trg)))

        # Revert classes for generator
        fake_domains_y_trg_for_gen = variable(torch.zeros(len(src)))
        fake_domains_y_src_for_gen = variable(torch.zeros(len(trg)))

        discr_src_loss_on_true = self.adv_criterion(true_domains_preds_src, true_domains_y_src)
        discr_src_loss_on_fake = self.adv_criterion(fake_domains_preds_src, fake_domains_y_src)
        discr_trg_loss_on_true = self.adv_criterion(true_domains_preds_trg, true_domains_y_trg)
        discr_trg_loss_on_fake = self.adv_criterion(fake_domains_preds_trg, fake_domains_y_trg)
        discr_src_loss = discr_src_loss_on_true + discr_src_loss_on_fake
        discr_trg_loss = discr_trg_loss_on_true + discr_trg_loss_on_fake

        # Uff, ok. Let's compute losses for our generators
        gen_src_to_trg_loss = self.adv_criterion(fake_domains_preds_trg, fake_domains_y_trg_for_gen)
        gen_trg_to_src_loss = self.adv_criterion(fake_domains_preds_src, fake_domains_y_src_for_gen)

        translator_loss += (gen_src_to_trg_loss + gen_trg_to_src_loss) * self.generator_loss_coef()

        ### Update weights ###
        # Translator update
        self.translator_optimizer.zero_grad()
        translator_loss.backward(retain_graph=True)
        self.translator_optimizer.step()
        
        # Discriminators updates
        self.discriminator_src_optimizer.zero_grad()
        self.discriminator_trg_optimizer.zero_grad()
        discr_src_loss.backward()
        discr_trg_loss.backward()
        self.discriminator_src_optimizer.step()
        self.discriminator_trg_optimizer.step()

        # Saving metrics
        self.train_scores['discr_src_loss_on_true'].append(discr_src_loss_on_true.data[0])
        self.train_scores['discr_src_loss_on_fake'].append(discr_src_loss_on_fake.data[0])
        self.train_scores['discr_trg_loss_on_true'].append(discr_trg_loss_on_true.data[0])
        self.train_scores['discr_trg_loss_on_fake'].append(discr_trg_loss_on_fake.data[0])
        self.train_scores['gen_src_to_trg_loss'].append(gen_src_to_trg_loss.data[0])
        self.train_scores['gen_trg_to_src_loss'].append(gen_trg_to_src_loss.data[0])
        self.train_scores['dae_src_loss'].append(dae_src_loss.data[0])
        self.train_scores['dae_trg_loss'].append(dae_trg_loss.data[0])

    def validate_bleu(self, val_data, return_results=False, beam_size=4):
        #self._eval_mode()

        sources = []
        targets = []
        translations_src_to_trg = []
        translations_trg_to_src = []
        translations_src_to_trg_to_src = []
        translations_trg_to_src_to_trg = []

        for src, trg in val_data:
            src_to_trg = self.translator.translate_batch(
                src, max_len=self.max_len, beam_size=beam_size)
            trg_to_src = self.translator.translate_batch(
                trg, max_len=self.max_len, beam_size=beam_size,
                use_src_embs_in_decoder=True, use_trg_embs_in_encoder=True)

            src_to_trg_var = pad_to_longest([[constants.BOS] + s for s in src_to_trg], volatile=True)
            trg_to_src_var = pad_to_longest([[constants.BOS] + s for s in trg_to_src], volatile=True)

            src_to_trg_to_src = self.translator.translate_batch(
                src_to_trg_var, max_len=self.max_len, beam_size=beam_size,
                use_src_embs_in_decoder=True, use_trg_embs_in_encoder=True)
            trg_to_src_to_trg = self.translator.translate_batch(
                trg_to_src_var, max_len=self.max_len, beam_size=beam_size)

            sources += token_ids_to_sents(src, self.vocab)
            targets += token_ids_to_sents(trg, self.vocab)

            translations_src_to_trg += token_ids_to_sents(src_to_trg, self.vocab)
            translations_trg_to_src += token_ids_to_sents(trg_to_src, self.vocab)

            translations_src_to_trg_to_src += token_ids_to_sents(src_to_trg_to_src, self.vocab)
            translations_trg_to_src_to_trg += token_ids_to_sents(trg_to_src_to_trg, self.vocab)

        bleu_src_to_trg_to_src = compute_bleu_for_sents(translations_src_to_trg_to_src, sources)
        bleu_trg_to_src_to_trg = compute_bleu_for_sents(translations_trg_to_src_to_trg, targets)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write('Epochs done: {}. Iters done: {}\n'.format(self.num_epochs_done, self.num_iters_done))
                f.write('[src->trg->src] BLEU: {}. [trg->src->trg] BLEU: {}.\n'.format(bleu_src_to_trg_to_src, bleu_trg_to_src_to_trg))

                f.write('##### SRC->TRG translations #####\n')
                for i, src_to_trg in enumerate(translations_src_to_trg):
                    f.write('Source: ' + sources[i] + '\n')
                    f.write('Result: ' + src_to_trg + '\n')

                if self.should_log_trg_to_src:
                    f.write('\n')
                    f.write('##### TRG->SRC translations #####\n')

                    for i, trg_to_src in enumerate(translations_trg_to_src):
                        f.write('Source: ' + targets[i] + '\n')
                        f.write('Result: ' + trg_to_src + '\n')

                f.write('\n===============================================================================\n')

        if return_results:
            scores = (bleu_src_to_trg_to_src, bleu_trg_to_src_to_trg)
            translations = {
                'sources': sources,
                'targets': targets,
                'translations_src_to_trg': translations_src_to_trg,
                'translations_trg_to_src': translations_trg_to_src,
                'translations_src_to_trg_to_src': translations_src_to_trg_to_src,
                'translations_trg_to_src_to_trg': translations_trg_to_src_to_trg
            }

            return scores, translations
        else:
            self.val_scores['bleu_src_to_trg_to_src'].append(bleu_src_to_trg_to_src)
            self.val_scores['bleu_trg_to_src_to_trg'].append(bleu_trg_to_src_to_trg)

    def plot_scores(self):
        clear_output(True)

        losses_to_display = [
            ('dae_src_loss', 'dae_trg_loss', 221),
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

        self._plot_val_scores()

    def _plot_val_scores(self):
        if not self.val_scores['bleu_src_to_trg_to_src']: return

        plt.figure(figsize=[16,4])

        src, trg = 'bleu_src_to_trg_to_src', 'bleu_trg_to_src_to_trg'
        val_iters = np.arange(len(self.val_scores[src])) * (self.num_iters_done / len(self.val_scores[src]))

        plt.title('Val translation BLEU')
        plt.plot(val_iters, self.val_scores[src], label=SCORES_TITLES[src])
        plt.plot(val_iters, self.val_scores[trg], label=SCORES_TITLES[trg])
        plt.grid()
        plt.legend()

        plt.show()

    def _run_dae(self, src_noised, trg_noised, src, trg):
        # Computing translation for ~src->src and ~trg->trg autoencoding tasks
        preds_src = self.translator(src_noised, src, use_src_embs_in_decoder=True)
        preds_trg = self.translator(trg_noised, trg, use_trg_embs_in_encoder=True)

        # Computing losses
        dae_loss_src = self.reconstruction_criterion(preds_src, src[:, 1:].contiguous().view(-1))
        dae_loss_trg = self.reconstruction_criterion(preds_trg, trg[:, 1:].contiguous().view(-1))

        return dae_loss_src, dae_loss_trg

    def _train_mode(self):
        if self.is_translator_shared:
            self.translator.train()
        else:
            raise NotImplemented

        if self.should_discriminate_translations:
            self.discriminator_src.train()
            self.discriminator_trg.train()
        else:
            raise NotImplemented

    def _eval_mode(self):
        if self.is_translator_shared:
            self.translator.eval()
        else:
            raise NotImplemented

        if self.should_discriminate_translations:
            self.discriminator_src.eval()
            self.discriminator_trg.eval()
        else:
            raise NotImplemented
            
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
