import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.data import Field, Dataset, Example, BucketIterator
from firelab import BaseTrainer
from firelab.utils.training_utils import cudable
from sklearn.model_selection import train_test_split

from src.models import RNNClassifier


class ClassifierTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassifierTrainer, self).__init__(config)

        self.losses['val_loss'] = [] # Here we'll write history for early stopping

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        domain_x_data_path = os.path.join(project_path, self.config.data.domain_x)
        domain_y_data_path = os.path.join(project_path, self.config.data.domain_y)

        with open(domain_x_data_path) as f: domain_x = f.read().splitlines()
        with open(domain_y_data_path) as f: domain_y = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        fields = [('domain_x', text), ('domain_y', text)]
        examples = [Example.fromlist([m,o], fields) for m,o in zip(domain_x, domain_y)]
        train_exs, val_exs = train_test_split(examples, test_size=self.config.val_set_size,
                                              random_state=self.config.random_seed)

        self.train_ds, self.val_ds = Dataset(train_exs, fields), Dataset(val_exs, fields)
        text.build_vocab(self.train_ds, max_size=self.config.hp.get('max_vocab_size'))

        self.vocab = text.vocab
        self.train_dataloader = BucketIterator(
            self.train_ds, batch_size, repeat=False, device=self.config.device_name)
        self.val_dataloader = BucketIterator(
            self.val_ds, batch_size, repeat=False, shuffle=False, device=self.config.device_name)

    def init_models(self):
        self.classifier = cudable(RNNClassifier(self.config.hp.model_size, self.vocab))

    def init_criterions(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def init_optimizers(self):
        self.optim = Adam(self.classifier.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        loss, acc = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Loss/train', loss.item(), self.num_iters_done)
        self.writer.add_scalar('Acc/train', acc, self.num_iters_done)

    def loss_on_batch(self, batch):
        preds_x = self.classifier(batch.domain_x)
        preds_y = self.classifier(batch.domain_y)

        loss_x = self.criterion(preds_x, torch.ones_like(preds_x))
        loss_y = self.criterion(preds_y, torch.zeros_like(preds_y))

        loss = (loss_x + loss_y) / 2

        acc_x = ((preds_x > 0) == torch.ones_like(preds_x).byte()).float().sum() / len(preds_x)
        acc_y = ((preds_y <= 0) == torch.ones_like(preds_y).byte()).float().sum() / len(preds_y)
        acc = ((acc_x + acc_y) / 2).item()

        return loss, acc

    def validate(self):
        losses = []
        accs = []

        for batch in self.val_dataloader:
            loss, acc = self.loss_on_batch(batch)
            losses.append(loss.item())
            accs.append(acc)

        self.writer.add_scalar('Loss/val', np.mean(losses), self.num_iters_done)
        self.writer.add_scalar('Acc/val', np.mean(accs), self.num_iters_done)
        self.losses['val_loss'].append(np.mean(losses))
