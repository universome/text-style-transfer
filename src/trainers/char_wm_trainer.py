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

from src.models import RNNLM, FFN, RNNEncoder, RNNDecoder
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.bleu import compute_bleu_for_sents
from src.inference import simple_inference
from src.utils.data_utils import itos_many, char_tokenize, word_base
from src.morph import morph_chars_idx, MORPHS_SIZE


class CharWMTrainer(BaseTrainer):
    def __init__(self, config):
        super(CharWMTrainer, self).__init__(config)
        self.losses['val_loss'] = []

    def init_dataloaders(self):
        project_path = self.config.firelab.project_path
        data_path = os.path.join(project_path, self.config.data)

        with open(data_path) as f: trg = f.read().splitlines()
        src = [word_base(w, self.config.hp.word_base_size) for w in trg]

        field = Field(init_token='<bos>', eos_token='<eos>',
                     batch_first=True, tokenize=char_tokenize)
        fields = [('src', field), ('trg', field)]
        examples = [Example.fromlist(pair, fields) for pair in zip(src, trg)]

        dataset = Dataset(examples, fields)
        split_ratio = 1 - (self.config.val_set_size / len(trg))
        self.train_ds, self.val_ds = dataset.split(split_ratio=split_ratio)
        field.build_vocab(self.train_ds)

        self.vocab = field.vocab
        self.train_dataloader = data.BucketIterator(
            self.train_ds, self.config.hp.batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(
            self.val_ds, self.config.hp.batch_size, repeat=False)

    def init_models(self):
        size = self.config.hp.model_size

        self.merge_z = cudable(FFN([MORPHS_SIZE + size, size]))
        self.encoder = cudable(RNNEncoder(size, size, self.vocab))
        self.decoder = cudable(RNNDecoder(size, size, self.vocab))

    def init_criterions(self):
        self.criterion = cross_entropy_without_pads(self.vocab)

    def init_optimizers(self):
        self.params = chain(
            self.merge_z.parameters(),
            self.encoder.parameters(),
            self.decoder.parameters(),
        )
        self.optim = Adam(self.params, lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Loss/train', loss.item(), self.num_iters_done)

    def loss_on_batch(self, batch):
        z = self.encoder(batch.src)
        morphs = morph_chars_idx(batch.trg, self.vocab)
        morphs = cudable(torch.from_numpy(morphs).float())
        z = self.merge_z(torch.cat([z, morphs], dim=1))
        logits = self.decoder(z, batch.trg[:, :-1])
        loss = self.criterion(logits.view(-1, len(self.vocab)), batch.trg[:, 1:].contiguous().view(-1))

        return loss

    def validate(self):
        losses = [self.loss_on_batch(cudable(b)).item() for b in self.val_dataloader]

        self.writer.add_scalar('Loss/val', np.mean(losses), self.num_iters_done)
        self.losses['val_loss'].append(np.mean(losses))

        self.validate_inference()

    def validate_inference(self):
        generated = []
        gold = []
        conditions = []

        for batch in self.val_dataloader:
            batch = cudable(batch)
            # Trying to reconstruct from first 3 letters
            z = self.encoder(batch.src)
            morphs = morph_chars_idx(batch.trg, self.vocab)
            morphs = cudable(torch.from_numpy(morphs).float())
            z = self.merge_z(torch.cat([z, morphs], dim=1))
            embs = self.decoder.embed(batch.trg[:, :4])
            z = self.decoder.gru(embs, z.unsqueeze(0))[1].squeeze(0)
            preds = simple_inference(self.decoder, z, self.vocab, max_len=30)

            sources = batch.trg[:, :4].cpu().numpy().tolist()
            results = [s + p for s,p in zip(sources, preds)]
            results = itos_many(results, self.vocab, sep='')

            generated.extend(results)
            gold.extend(itos_many(batch.trg, self.vocab, sep=''))
            conditions.extend(itos_many(batch.src, self.vocab, sep=''))

        # Let's try to measure BLEU scores (although it's not valid for words)
        sents_generated = [' '.join(list(s[4:])) for s in generated]
        sents_gold = [' '.join(list(s[4:])) for s in gold]
        bleu = compute_bleu_for_sents(sents_generated, sents_gold)
        self.writer.add_scalar('BLEU', bleu, self.num_iters_done)

        texts = ['{} ({} | {}) => {}'.format(g[:3],c,g,p) for p,g,c in zip(generated, gold, conditions)]
        texts = random.sample(texts, 10) # Limiting amount of displayed text
        text = '\n\n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
