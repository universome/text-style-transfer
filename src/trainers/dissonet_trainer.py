import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils import cudable

from src.models.dissonet import RNNEncoder, RNNDecoder
from src.models import FFN
# from src.losses import compute_bleu_for_sents
# from src.utils.common import itos_many
# from src.models.utils import inference


class DissoNetTrainer(BaseTrainer):
    def __init__(self, config):
        super(DissoNetTrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.get('batch_size', 8)
        project_path = self.config['firelab']['project_path']
        modern_data_path = os.path.join(project_path, self.config['data']['modern'])
        original_data_path = os.path.join(project_path, self.config['data']['original'])

        with open(modern_data_path) as f: modern = f.read().splitlines()
        with open(original_data_path) as f: original = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        fields = [('modern', text), ('original', text)]
        examples = [Example.fromlist([m,o], fields) for m,o in zip(modern, original)]

        dataset = Dataset(examples, fields)
        self.train_ds, self.val_ds = dataset.split(split_ratio=[0.99, 0.01])
        text.build_vocab(self.train_ds)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size, repeat=False, shuffle=False)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, repeat=False, shuffle=False)

    def init_models(self):
        emb_size = self.config['hp'].get('emb_size')
        hid_size = self.config['hp'].get('hid_size')
        voc_size = len(self.vocab)

        # Defining models
        self.encoder = cudable(RNNEncoder(emb_size, hid_size, voc_size))
        self.decoder = cudable(RNNDecoder(emb_size, hid_size, voc_size))
        self.critic = cudable(FFN(hid_size // 2, 1, hid_size=hid_size))
        self.motivator = cudable(FFN(hid_size // 2, 1, hid_size=hid_size))

        self.init_criterions()

        # Defining optimizers
        lr = self.config['hp'].get('lr', 1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.motivator_optim = Adam(self.motivator.parameters(), lr=lr)
        self.ae_optim = Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)

    def init_criterions(self):
        # Reconstruction loss
        weights = cudable(torch.ones(len(self.vocab)))
        weights[self.vocab.stoi['<pad>']] = 0
        self.rec_criterion = nn.CrossEntropyLoss(weights)

        # Critic loss. Is similar to WGAN (but without lipschitz constraints)
        class CriticLoss(nn.Module):
            def __init__(self):
                super(CriticLoss, self).__init__()

            def forward(self, real, fake):
                return real.mean() - fake.mean()

        self.critic_criterion = CriticLoss()

        # Motivator loss
        self.motivator_criterion = nn.BCEWithLogitsLoss()


    def train_on_batch(self, batch):
        rec_loss, motivator_loss, critic_loss, ae_loss = self.loss_on_batch(batch)

        # Now we can make backward passes
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.ae_optim.zero_grad()
        self.motivator_optim.zero_grad()
        motivator_loss.backward(retain_graph=True)
        self.motivator_optim.step()
        self.ae_optim.step()

        self.ae_optim.zero_grad()
        ae_loss.backward(retain_graph=True)
        self.ae_optim.step()

        self.writer.add_scalar('Rec loss', rec_loss, self.num_iters_done)
        self.writer.add_scalar('Motivator loss', motivator_loss, self.num_iters_done)
        self.writer.add_scalar('Critic loss', critic_loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        # Computing codes we need
        style_modern, content_modern = self.encoder(batch.modern)
        style_original, content_original = self.encoder(batch.original)

        # Now we should merge back style and content for decoder
        hid_modern = torch.cat([style_modern, content_modern], dim=1)
        hid_original = torch.cat([style_original, content_original], dim=1)

        # Reconstructing
        recs_modern = self.decoder(hid_modern, batch.modern[:, :-1])
        recs_original = self.decoder(hid_original, batch.original[:, :-1])

        # Computing reconstruction loss
        rec_loss_modern = self.rec_criterion(recs_modern.view(-1, len(self.vocab)), batch.modern[:, 1:].contiguous().view(-1))
        rec_loss_original = self.rec_criterion(recs_original.view(-1, len(self.vocab)), batch.original[:, 1:].contiguous().view(-1))
        rec_loss = rec_loss_modern + rec_loss_original

        # Computing critic loss
        critic_modern_preds, critic_original_preds = self.critic(content_modern), self.critic(content_original)
        critic_loss = self.critic_criterion(critic_modern_preds, critic_original_preds)

        # Computing motivator loss
        motivator_logits_modern = self.motivator(style_modern)
        motivator_logits_original = self.motivator(style_original)
        motivator_loss_modern = self.motivator_criterion(motivator_logits_modern, torch.ones_like(motivator_logits_modern))
        motivator_loss_original = self.motivator_criterion(motivator_logits_original, torch.zeros_like(motivator_logits_original))
        motivator_loss = motivator_loss_modern + motivator_loss_original

        # Loss for encoder and decoder is threefold
        ae_loss = rec_loss + motivator_loss - critic_loss

        return rec_loss, motivator_loss, critic_loss, ae_loss

    def validate(self):
        return
        rec_losses = []

        for batch in self.val_dataloader:
            rec_losses.append(self.loss_on_batch(batch).item())

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.compute_val_bleu()

    def compute_val_bleu(self):
        """
        Performs inference on a val dataloader
        (computes predictions without teacher's forcing)
        """
        generated, originals = self.inference(self.val_dataloader)
        bleu = compute_bleu_for_sents(generated, originals)
        generated = ['[{}] => [{}]'.format(o,g) for o,g in zip(originals, generated)]
        text = '\n\n'.join(generated)

        self.writer.add_text('Generated examples', text, self.num_iters_done)
        self.writer.add_scalar('Validation BLEU', bleu, self.num_iters_done)

    def inference(self, dataloader):
        """
        Produces predictions for a given dataloader
        """
        seqs = []
        originals = []

        for batch in dataloader:
            inputs = cudable(batch.text)
            encodings = self.encoder(inputs)
            sentences = inference(self.decoder, encodings, self.vocab)

            seqs.extend(sentences)
            originals.extend(inputs.detach().cpu().numpy().tolist())

        return itos_many(seqs, self.vocab), itos_many(originals, self.vocab)
