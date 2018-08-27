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
from firelab.utils.data_utils import filter_sents_by_len
from sklearn.model_selection import train_test_split

from src.utils.data_utils import itos_many
from src.inference import inference
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.models.transformer import TransformerEmbedder, TransformerDecoder
from src.models.dissonet import RNNEncoder, RNNDecoder


class TransformerEmbedderTrainer(BaseTrainer):
    def __init__(self, config):
        super(TransformerEmbedderTrainer, self).__init__(config)

        self.losses['val_bleu'] = [] # Here we'll write history for early stopping

    def init_dataloaders(self):
        project_path = self.config.firelab.project_path
        data_path = os.path.join(project_path, self.config.data)
        with open(data_path) as f: lines = f.read().splitlines()

        lines = [s for s in lines if self.config.hp.min_len <= len(s.split()) <= self.config.hp.max_len]

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)

        examples = [Example.fromlist([s], [('text', text)]) for s in lines]
        dataset = Dataset(examples, [('text', text)])
        split_ratio = 1 - (self.config.val_set_size / len(lines))
        self.train_ds, self.val_ds = dataset.split(split_ratio=split_ratio)
        text.build_vocab(self.train_ds)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(self.train_ds, self.config.hp.batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(self.val_ds,
            self.config.hp.batch_size, repeat=False, shuffle=False, sort=False)

    def init_models(self):
        self.encoder = cudable(TransformerEmbedder(self.config.hp.n_vecs, self.config.hp.transformer, self.vocab))
        self.decoder = cudable(TransformerDecoder(self.config.hp.transformer, self.vocab))
        # self.encoder = cudable(RNNEncoder(512, 512, len(self.vocab)))
        # self.decoder = cudable(RNNDecoder(512, 512, len(self.vocab)))

    def init_criterions(self):
        self.criterion = cross_entropy_without_pads(self.vocab)

    def init_optimizers(self):
        self.optim = Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('train/loss', loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        embs = self.encoder(batch.text)
        recs = self.decoder(embs, batch.text[:, :-1])
        loss = self.criterion(recs.view(-1, len(self.vocab)), batch.text[:, 1:].contiguous().view(-1))

        return loss

    def validate(self):
        all_recs = []
        all_gold = []
        losses = []

        for batch in self.val_dataloader:
            loss = self.loss_on_batch(batch)
            embs = self.encoder(batch.text)

            recs = inference(self.decoder, embs, self.vocab)

            recs = itos_many(recs, self.vocab)
            gold = itos_many(batch.text, self.vocab)

            all_recs.extend(recs)
            all_gold.extend(gold)
            losses.append(loss.item())

        bleu = compute_bleu_for_sents(all_recs, all_gold)

        self.writer.add_scalar('VAL/loss', np.mean(losses), self.num_iters_done)
        self.writer.add_scalar('VAL/BLEU', bleu, self.num_iters_done)

        self.losses['val_bleu'].append(bleu)
