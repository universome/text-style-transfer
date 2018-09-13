import os
import math
import random
import pickle
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Example, Dataset
from firelab import BaseTrainer
from firelab.utils.training_utils import cudable

from src.models import RNNLM
from src.utils.data_utils import itos_many, char_tokenize
from src.optims.triangle_adam import TriangleAdam


class CharRNNTrainer(BaseTrainer):
    def __init__(self, config):
        super(CharRNNTrainer, self).__init__(config)

    def init_dataloaders(self):
        print('Initializing dataloaders')

        project_path = self.config.firelab.project_path
        data_path_train = os.path.join(project_path, self.config.data.train)
        data_path_val = os.path.join(project_path, self.config.data.val)

        data_train = open(data_path_train).read().splitlines()
        data_val = open(data_path_val).read().splitlines()[:self.config.val_set_size]

        self.eos = '|'
        field = Field(eos_token=self.eos, batch_first=True, tokenize=char_tokenize)

        train_examples = [Example.fromlist([self.eos.join(data_train)], [('text', field)])]
        val_examples = [Example.fromlist([s], [('text', field)]) for s in data_val]

        self.train_ds = Dataset(train_examples, [('text', field)])
        self.val_ds = Dataset(val_examples, [('text', field)])

        field.build_vocab(self.train_ds)

        self.vocab = field.vocab
        self.train_dataloader = data.BPTTIterator(self.train_ds,
            self.config.hp.batch_size, self.config.hp.batch_len, repeat=False)
        self.val_dataloader = data.BucketIterator(
            self.val_ds, 1, shuffle=False, repeat=False)

        print('Dataloaders initialized!')

    def init_models(self):
        self.lm = cudable(RNNLM(self.config.hp.model_size,
            self.vocab, n_layers=self.config.hp.n_layers))

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss()

    def init_optimizers(self):
        self.optim = TriangleAdam(self.lm.parameters(), self.config.hp.optim)

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train loss', loss.item(), self.num_iters_done)

    def loss_on_batch(self, batch):
        batch.text = cudable(batch.text).transpose(0,1)
        z = cudable(torch.zeros(batch.batch_size, self.config.hp.model_size))
        preds = self.lm(z, batch.text[:, :-1])
        loss = self.criterion(preds.view(-1, len(self.vocab)), batch.text[:, 1:].contiguous().view(-1))

        return loss

    def validate(self):
        sources = []
        generated = []
        batches = list(self.val_dataloader)

        for batch in random.sample(batches, self.config.display_k_val_examples):
            src = cudable(batch.text)
            results = self.lm.inference(src.squeeze(), self.vocab, eos_token=self.eos, max_len=250)
            results = itos_many([results], self.vocab, sep='')

            generated.extend(results)
            sources.extend(itos_many(batch.text, self.vocab, sep=''))

        texts = ['`{} => {}`'.format(s,g) for s,g in zip(sources, generated)]
        text = '\n\n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
