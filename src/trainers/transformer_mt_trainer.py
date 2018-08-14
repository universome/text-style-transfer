import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data, datasets
from firelab import BaseTrainer
from firelab.utils import cudable

from src.models.transformer import Transformer
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.inference import inference


class TransformerMTTrainer(BaseTrainer):
    """
    Machine translation with Transformer to test implementation
    """
    def __init__(self, config):
        super(TransformerMTTrainer, self).__init__(config)

        self.losses['val_bleu'] = []

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_path_train = os.path.join(project_path, self.config.data.train)
        data_path_val = os.path.join(project_path, self.config.data.val)

        src = data.Field(batch_first=True)
        trg = data.Field(batch_first=True)

        mt_train = datasets.TranslationDataset(
            path=data_path_train, exts=('.en', '.fr'), fields=(src, trg))
        mt_val = datasets.TranslationDataset(
            path=data_path_val, exts=('.en', '.fr'), fields=(src, trg))

        src.build_vocab(mt_train)
        trg.build_vocab(mt_train)

        self.vocab_src = src.vocab
        self.vocab_trg = trg.vocab

        self.train_dataloader = data.BucketIterator(mt_train, batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(mt_val, batch_size, repeat=False)

    def init_models(self):
        self.transformer = cudable(Transformer(self.config.hp.transformer, self.vocab_src, self.vocab_trg))

    def init_criterions(self):
        self.criterion = cross_entropy_without_pads(self.vocab_trg)

    def init_optimizers(self):
        self.optim = Adam(self.transformer.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        rec_loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        rec_loss.backward()
        self.optim.step()

        self.writer.add_scalar('train/rec_loss', rec_loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        recs = self.transformer(batch.src, batch.trg)
        targets = batch.trg[:, 1:].contiguous().view(-1)
        rec_loss = self.criterion(recs.view(-1, len(self.vocab_trg)), targets)

        return rec_loss

    def validate(self):
        rec_losses = []
        bleus = []

        for batch in self.val_dataloader:
            # CE loss
            rec_loss = self.loss_on_batch(batch)
            rec_losses.append(rec_loss.item())

            # BLEU
            encs, mask = self.transformer.encoder(batch.src)
            preds = inference(self.transformer.decoder, encs, self.vocab_trg, mask)
            preds = itos_many(preds, self.vocab_trg)
            gold = itos_many(batch.trg, self.vocab_trg)
            bleu = compute_bleu_for_sents(preds, gold)
            bleus.append(bleu)

        self.writer.add_scalar('val/rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.writer.add_scalar('val/bleu', np.mean(bleus), self.num_iters_done)
        self.losses['val_bleu'].append(np.mean(bleus))
