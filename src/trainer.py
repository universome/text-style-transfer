import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd

use_cuda = torch.cuda.is_available()

LOSSES_TITLES = {
    'ae_loss_src': '[src] lang AE loss',
    'ae_loss_trg': '[trg] lang AE loss',
    'loss_bt_src': '[src] lang back-translation loss',
    'loss_bt_trg': '[trg] lang back-translation loss',
    'discr_loss_src': '[src] lang discriminator loss',
    'discr_loss_trg': '[trg] lang discriminator loss',
    'gen_loss_src': '[src] lang generator loss',
    'gen_loss_trg': '[trg] lang generator loss'
}


class UMTTrainer:
    def __init__(translator, discriminator,
        translator_optimizer, dicsrriminator_optimizer, config):

        self.translator = translator
        self.discriminator = discriminator
        self.translator = translator_optimizer
        self.discriminator = discriminator_optimizer

        self.num_iters_done = 0
        self.num_epochs_done = 0
        self.max_num_epochs = config.get('max_num_epochs', 100)
        self.start_bt_from_epoch = config.get('start_bt_from_epoch', 1)

        self.losses = {
            'ae_loss_src': [],
            'ae_loss_trg': [],
            'loss_bt_src': [],
            'loss_bt_trg': [],
            'discr_loss_src': [],
            'discr_loss_trg': [],
            'gen_loss_src': [],
            'gen_loss_trg': []
        }

    def run_training(training_data, visualize_losses=False):
        should_continue = True

        while self.num_epochs_done < self.max_num_epochs and should_continue:
            for batch in tqdm(training_data, leave=False):
                try:
                    self.train_on_batch(batch)
                    self.num_iters_done += 1
                    if visualize_losses: self.visualize_losses()
                except KeyboardInterrupt:
                    should_continue = False
                    break

        self.num_epochs_done += 1

    def train_on_batch(batch):
        src_noised, trg_noised, src, trg = batch

        # Resetting gradients
        self.transformer_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        should_backtranslate = self.num_epochs_done >= self.start_bt_from_epoch

        encodings = self._train_ae(src_noised, trg_noised, src, trg)
        if should_backtranslate: self._train_bt(src, trg)
        domains_predictions = self._train_discriminator(*encodings)
        self._train_generator(*domains_predictions)

    def _train_ae(src_noised, trg_noised, src, trg):
        ### Training autoencoder ###
        self.translator.train()
        # Computing translation for ~src->src and ~trg->trg autoencoding tasks
        print('Training discriminator')
        print('Computing predictions')
        preds_src, encodings_src = self.translator(src_noised, src, return_encodings=True, use_src_embs_in_decoder=True)
        preds_trg, encodings_trg = self.translator(trg_noised, trg, return_encodings=True, use_trg_embs_in_encoder=True)

        print('Computing losses')
        ae_loss_src = ae_criterion_src(preds_src, src[:, 1:].contiguous().view(-1))
        ae_loss_trg = ae_criterion_trg(preds_trg, trg[:, 1:].contiguous().view(-1))

        print('Computing gradients')
        ae_loss_src.backward(retain_graph=True)
        ae_loss_trg.backward(retain_graph=True)

        self.losses['ae_loss_src'].append(ae_loss_src.data[0])
        self.losses['ae_loss_trg'].append(ae_loss_trg.data[0])

    def _train_bt(src, trg):
        ### Training translator ###
        print('Training translator')
        self.translator.eval()
        # Get translations for backtranslation
        print('Computing back-translations')
        bt_trg, *_ = self.translator.translate_batch(src, beam_size=2, max_len=10)
        bt_src, *_ = self.translator.translate_batch(trg, use_trg_embs_in_encoder=True, use_src_embs_in_decoder=True, beam_size=2, max_len=10)

        bt_trg = Variable(torch.LongTensor(bt_trg))
        bt_src = Variable(torch.LongTensor(bt_src))

        # We are given n-best translations. Let's pick the best one
        bt_trg = bt_trg[:,0,:]
        bt_src = bt_src[:,0,:]

        if use_cuda:
            bt_trg = bt_trg.cuda()
            bt_src = bt_src.cuda()

        # Computing predictions for back-translated sentences
        self.translator.train()
        print('Computing predictions (translations of back-translations)')
        bt_src_preds = self.translator(bt_trg, src, use_trg_embs_in_encoder=True, use_src_embs_in_decoder=True)
        bt_trg_preds = self.translator(bt_src, trg)

        print('Computing losses')
        loss_bt_src = translation_criterion_trg_to_src(bt_src_preds, src[:, 1:].contiguous().view(-1))
        loss_bt_trg = translation_criterion_src_to_trg(bt_trg_preds, trg[:, 1:].contiguous().view(-1))

        print('Computing gradients')
        loss_bt_src.backward(retain_graph=True)
        loss_bt_trg.backward(retain_graph=True)

        print('Updating weights')
        self.transformer_optimizer.step()

        self.transformer_optimizer.zero_grad()

        self.losses['loss_bt_src'].append(loss_bt_src.data[0])
        self.losses['loss_bt_trg'].append(gen_loss_trg.data[0])

    def _train_discriminator(encodings_src, encodings_trg):
        ### Training discriminator ###
        print('Training discriminator')
        print('Computing predictions')
        domains_preds_src = discriminator(encodings_src.view(-1, 512))
        domains_preds_trg = discriminator(encodings_trg.view(-1, 512))

        # Generating targets for discriminator
        true_domains_src = Variable(torch.Tensor([0] * len(domains_preds_src)))
        true_domains_trg = Variable(torch.Tensor([1] * len(domains_preds_trg)))

        if use_cuda:
            true_domains_src = true_domains_src.cuda()
            true_domains_trg = true_domains_trg.cuda()

        # True domains for discriminator loss
        print('Computing losses')
        discr_loss_src = adv_criterion(domains_preds_src, true_domains_src)
        discr_loss_trg = adv_criterion(domains_preds_trg, true_domains_trg)

        print('Computing gradients')
        discr_loss_src.backward(retain_graph=True)
        discr_loss_trg.backward(retain_graph=True)

        print('Updating parameters')
        self.discriminator_optimizer.step()

        # Cleaning up gradients
        self.transformer_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        self.losses['discr_loss_src'].append(discr_loss_src.data[0])
        self.losses['discr_loss_trg'].append(discr_loss_trg.data[0])

        return domains_preds_src, domains_preds_trg

    def _train_generator(domains_preds_src, domains_preds_trg):
        src_noised, trg_noised, src, trg = batch

        fake_domains_src = Variable(torch.Tensor([1] * len(domains_preds_src)))
        fake_domains_trg = Variable(torch.Tensor([0] * len(domains_preds_trg)))

        if use_cuda:
            fake_domains_src = fake_domains_src.cuda()
            fake_domains_trg = fake_domains_trg.cuda()

        ### Training generator ###
        print('Training generator')
        print('Computing losses')
        # Faking domains for generator loss
        gen_loss_src = adv_criterion(domains_preds_src, fake_domains_src)
        gen_loss_trg = adv_criterion(domains_preds_trg, fake_domains_trg)

        print('Computing gradients')
        gen_loss_src.backward(retain_graph=True)
        gen_loss_trg.backward(retain_graph=True)

        print('Updating parameters')
        self.transformer_optimizer.step()

        self.losses['gen_loss_src'].append(gen_loss_src.data[0])
        self.losses['gen_loss_trg'].append(gen_loss_trg.data[0])

    def visualize_losses():
        clear_output(True)

        losses_pairs = [
            ('ae_loss_src', 'ae_loss_trg'),
            ('loss_bt_src', 'loss_bt_trg'),
            ('discr_loss_src', 'discr_loss_trg'),
            ('gen_loss_src', 'gen_loss_trg')
        ]

        for src, trg in losses_pair:
            if len(self.losses[src]) == 0 or len(self.losses[trg]) == 0:
                continue

            figure, axes = plt.subplots(2, sharey=True)
            figure.set_figwidth(16)
            figure.set_figheight(6)

            plt.subplot(121)
            plt.title(LOSSES_TITLES[src])
            plt.plot(self.losses[src])
            plt.plot(pd.DataFrame(self.losses[src]).ewm(span=50))
            plt.grid()

            plt.subplot(122)
            plt.title(LOSSES_TITLES[trg])
            plt.plot(self.losses[trg])
            plt.plot(pd.DataFrame(self.losses[trg]).ewm(span=50))
            plt.grid()

        plt.show()
