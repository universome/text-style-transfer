import os
import math
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Example, Dataset
from firelab import BaseTrainer
from firelab.utils.training_utils import cudable

from src.models import ConditionalLM
from src.utils.data_utils import itos_many, char_tokenize, split_in_batches
from src.optims.triangle_adam import TriangleAdam
from src.inference import InferenceState


class ConditionalCharRNNTrainer(BaseTrainer):
    def __init__(self, config):
        super(ConditionalCharRNNTrainer, self).__init__(config)

    def init_dataloaders(self):
        print('Initializing dataloaders')

        self.eos = '|'
        project_path = self.config.firelab.project_path
        data_path_train_x = os.path.join(project_path, self.config.data.train.domain_x)
        data_path_train_y = os.path.join(project_path, self.config.data.train.domain_y)
        data_path_val = os.path.join(project_path, self.config.data.val)

        data_val = open(data_path_val).read().splitlines()[:self.config.val_set_size]
        data_train_x = open(data_path_train_x).read().splitlines()
        data_train_y = open(data_path_train_y).read().splitlines()

        # Concatenating into text
        data_train_x = self.eos.join(data_train_x)
        data_train_y = self.eos.join(data_train_y)
        data_train_all = data_train_x + self.eos + data_train_y

        # Splitting larger dataset into groups
        if len(data_train_y) < len(data_train_x):
            self.data_x_groups = split_in_batches(data_train_x, len(data_train_y))
            self.data_y_groups = [data_train_y]
        elif len(data_train_x) < len(data_train_y):
            self.data_x_groups = [data_train_x]
            self.data_y_groups = split_in_batches(data_train_y, len(data_train_x))
        else:
            self.data_x_groups = [data_train_x]
            self.data_y_groups = [data_train_y]

        self.field = Field(eos_token=self.eos, batch_first=True, tokenize=char_tokenize)

        examples = [Example.fromlist([data_train_all], [('text', self.field)])]
        train_ds = Dataset(examples, [('text', self.field)])
        self.field.build_vocab(train_ds)

        val_examples = [Example.fromlist([s], [('text', self.field)]) for s in data_val]
        val_ds = Dataset(val_examples, [('text', self.field)])

        self.vocab = self.field.vocab
        self.build_train_dataloader()
        self.val_dataloader = data.BucketIterator(
            val_ds, self.config.display_k_val_examples, shuffle=False, repeat=True)
        self.val_iter = iter(self.val_dataloader)

        print('Dataloaders initialized!')

    def on_epoch_done(self):
        self.build_train_dataloader()

    def build_train_dataloader(self):
        print('Building train dataloader...')
        data_x = random.sample(self.data_x_groups, 1)[0]
        data_y = random.sample(self.data_y_groups, 1)[0]

        examples_x = [Example.fromlist([data_x], [('text', self.field)])]
        examples_y = [Example.fromlist([data_y], [('text', self.field)])]

        ds_x = Dataset(examples_x, [('text', self.field)])
        ds_y = Dataset(examples_y, [('text', self.field)])

        dataloader_x = data.BPTTIterator(
            ds_x, self.config.hp.batch_size, self.config.hp.batch_len, repeat=False)
        dataloader_y = data.BPTTIterator(
            ds_y, self.config.hp.batch_size, self.config.hp.batch_len, repeat=False)

        self.train_dataloader = list(zip(dataloader_x, dataloader_y))
        print('Train dataloader has been built!')

    def init_models(self):
        self.lm = cudable(ConditionalLM(self.config.hp.model_size, self.vocab))

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss()

    def init_optimizers(self):
        self.optim = TriangleAdam(self.lm.parameters(), self.config.hp.optim)

    def train_on_batch(self, batch):
        loss, info = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train/loss_x', info['loss_x'], self.num_iters_done)
        self.writer.add_scalar('Train/loss_y', info['loss_y'], self.num_iters_done)

    def loss_on_batch(self, batch):
        domain_x, domain_y = batch[0].text, batch[1].text
        domain_x = cudable(domain_x).transpose(0,1)
        domain_y = cudable(domain_y).transpose(0,1)

        z_x = cudable(torch.zeros(2, len(domain_x), self.config.hp.model_size))
        z_y = cudable(torch.zeros(2, len(domain_y), self.config.hp.model_size))

        preds_x = self.lm(z_x.contiguous(), domain_x[:, :-1], style=0)
        preds_y = self.lm(z_y.contiguous(), domain_y[:, :-1], style=1)

        loss_x = self.criterion(preds_x.view(-1, len(self.vocab)), domain_x[:, 1:].contiguous().view(-1))
        loss_y = self.criterion(preds_y.view(-1, len(self.vocab)), domain_y[:, 1:].contiguous().view(-1))

        info = {'loss_x': loss_x.item(), 'loss_y': loss_y.item()}
        loss = (loss_x + loss_y) / 2

        return loss, info

    def validate(self):
        sources = []
        generated = []

        batch = next(self.val_iter)
        z = cudable(torch.zeros(2, len(batch.text), self.config.hp.model_size))
        z = self.lm(z, cudable(batch.text), style=1, return_z=True)[1]

        results = InferenceState({
            'model': self.lm,
            'vocab': self.vocab,
            'inputs': z,
            'eos_token': self.eos,
            'bos_token': self.eos, # We start decoding from eos. TODO: looks like a hack
            'max_len': 250,
            'inputs_batch_first': False,
            'kwargs': {'style': 1}
        }).inference()
        results = itos_many(results, self.vocab, sep='')
        results = [''.join(s for s in sent if s != self.eos) for sent in results]

        generated.extend(results)
        sources.extend(itos_many(batch.text, self.vocab, sep=''))

        texts = ['`{} => {}`'.format(s,g) for s,g in zip(sources, generated)]
        text = '\n\n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
