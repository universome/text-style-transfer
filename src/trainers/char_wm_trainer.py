import os
import math
import random
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab import BaseTrainer
from firelab.utils.training_utils import cudable

from src.models import RNNLM
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.bleu import compute_bleu_for_sents
from src.inference import inference
from src.utils.data_utils import itos_many


class CharWMTrainer(BaseTrainer):
    def __init__(self, config):
        super(CharWMTrainer, self).__init__(config)
        self.losses['val_loss'] = []

    def init_dataloaders(self):
        project_path = self.config.firelab.project_path
        data_path = os.path.join(project_path, self.config.data)

        with open(data_path) as f: lines = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>',
                     batch_first=True, tokenize=lambda s: list(s))
        examples = [Example.fromlist([s], [('text', text)]) for s in lines]

        dataset = Dataset(examples, [('text', text)])
        split_ratio = 1 - (self.config.val_set_size / len(lines))
        self.train_ds, self.val_ds = dataset.split(split_ratio=split_ratio)
        text.build_vocab(self.train_ds)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(
            self.train_ds, self.config.batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(
            self.val_ds, self.config.batch_size, repeat=False)

    def init_models(self):
        self.lm = cudable(RNNLM(self.config.hp.model_size, self.vocab))

    def init_criterions(self):
        self.criterion = cross_entropy_without_pads(self.vocab)

    def init_optimizers(self):
        self.optim = Adam(self.lm.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Loss/train', loss.item(), self.num_iters_done)

    def loss_on_batch(self, batch):
        z = cudable(torch.zeros(batch.batch_size, self.config.hp.model_size))
        preds = self.lm(z, batch.text[:, :-1])
        loss = self.criterion(preds.view(-1, len(self.vocab)), batch.text[:, 1:].contiguous().view(-1))

        return loss

    def validate(self):
        losses = [self.loss_on_batch(b).item() for b in self.val_dataloader]

        self.writer.add_scalar('Loss/val', np.mean(losses), self.num_iters_done)
        self.losses['val_loss'].append(np.mean(losses))

        self.validate_inference()

    def validate_inference(self):
        generated = []
        gold = []

        for batch in self.val_dataloader:
            # Trying to reconstruct from first 3 letters
            embs = self.lm.embed(batch.text[:, :4])
            _, z = self.lm.gru(embs)
            z = z.squeeze()
            preds = inference(self.lm, z, self.vocab, max_len=30)

            sources = batch.text[:, :4].cpu().numpy().tolist()
            results = [s + p for s,p in zip(sources, preds)]
            results = itos_many(results, self.vocab, sep='')

            generated.extend(results)
            gold.extend(itos_many(batch.text, self.vocab, sep=''))

        # Let's try to measure BLEU scores (although it's not valid for words)
        sents_generated = [' '.join(list(s[4:])) for s in generated]
        sents_gold = [' '.join(list(s[4:])) for s in gold]
        bleu = compute_bleu_for_sents(sents_generated, sents_gold)
        self.writer.add_scalar('BLEU', bleu, self.num_iters_done)

        texts = ['{} ({}) => {}'.format(g[:3],g,p) for p,g in zip(generated, gold)]
        texts = random.sample(texts, 10) # Limiting amount of displayed text
        text = '\n\n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
