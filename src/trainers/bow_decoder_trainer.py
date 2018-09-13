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
from firelab.utils.training_utils import cudable

from src.models import RNNEncoder, RNNDecoder
from src.models import FFN
from src.utils.data_utils import itos_many
from src.losses.bleu import compute_bleu_for_sents
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.gan_losses import WCriticLoss
from src.inference import simple_inference


class BowDecoderTrainer(BaseTrainer):
    """
    Can we decode bag-of-words sentence?
    """
    def __init__(self, config):
        super(BowDecoderTrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.get('batch_size', 8)
        project_path = self.config['firelab']['project_path']
        data_path = os.path.join(project_path, self.config['data'])

        with open(data_path) as f: lines = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)
        examples = [Example.fromlist([s], [('text', text)]) for s in lines]

        dataset = Dataset(examples, [('text', text)])
        self.train_ds, self.val_ds = dataset.split(split_ratio=[0.99, 0.01])
        text.build_vocab(self.train_ds)

        self.vocab = text.vocab
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size, repeat=False, shuffle=False)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, repeat=False, shuffle=False)

    def init_models(self):
        emb_size = self.config['hp'].get('emb_size')
        hid_size = self.config['hp'].get('hid_size')
        voc_size = len(self.vocab)

        self.decoder = cudable(RNNDecoder(emb_size, hid_size, voc_size))
        self.embed = cudable(FFN(voc_size, 1, hid_size=hid_size, output_size=emb_size))

    def init_criterions(self):
        self.rec_criterion = cross_entropy_without_pads(self.vocab)

    def init_optimizers(self):
        lr = self.config['hp'].get('lr', 1e-4)

        self.optim = Adam(chain(self.embed.parameters(), self.decoder.parameters()), lr=lr)

    def train_on_batch(self, batch):
        rec_loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        rec_loss.backward()
        self.optim.step()

        self.writer.add_scalar('Rec loss', rec_loss, self.num_iters_done)

    def loss_on_batch(self, batch):
        inputs = sents_to_bow(batch.text, len(self.vocab))
        embs = self.embed(inputs)
        recs = self.decoder(embs, batch.text[:, :-1])
        targets = batch.text[:, 1:].contiguous().view(-1)

        rec_loss = self.rec_criterion(recs.view(-1, len(self.vocab)), targets)

        return rec_loss

    def validate(self):
        rec_losses = []
        bleus = []

        for batch in self.val_dataloader:
            # CE loss
            rec_loss = self.loss_on_batch(batch)
            rec_losses.append(rec_loss.item())

            # BLEU
            embs = self.embed(sents_to_bow(batch.text, len(self.vocab)))
            preds = simple_inference(self.decoder, embs, self.vocab)
            preds = itos_many(preds, self.vocab)
            gold = itos_many(batch.text[:, 1:], self.vocab)
            bleu = compute_bleu_for_sents(preds, gold)
            bleus.append(bleu)

        self.writer.add_scalar('val_rec_loss', np.mean(rec_losses), self.num_iters_done)
        self.writer.add_scalar('val_bleu', np.mean(bleus), self.num_iters_done)


def sents_to_bow(sents, voc_size):
    """
    Converts normal sentence (list of token ids) into BoW vector
    TODO: remove for loop :|
    """
    batch_size = sents.size(0)
    out = cudable(torch.zeros(batch_size, voc_size))

    for col in sents.t():
        out[torch.arange(batch_size).long(), col.long()] = 1

    return out
