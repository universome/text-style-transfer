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
from sklearn.model_selection import train_test_split

from src.models import FFN
from src.models.dissonet import RNNEncoder, RNNDecoder
from src.losses.ce_without_pads import cross_entropy_without_pads
from src.losses.bleu import compute_bleu_for_sents
from src.inference import inference
from src.utils.data_utils import itos_many, char_tokenize
from src.morph import morph_chars_idx, MORPHS_SIZE


class WordFillingTrainer(BaseTrainer):
    def __init__(self, config):
        super(WordFillingTrainer, self).__init__(config)
        self.losses['val_loss'] = []

    def init_dataloaders(self):
        project_path = self.config.firelab.project_path
        data_path_src = os.path.join(project_path, self.config.data.src)
        data_path_trg = os.path.join(project_path, self.config.data.trg)

        with open(data_path_src) as f: src = f.read().splitlines()
        with open(data_path_trg) as f: trg = f.read().splitlines()

        text = Field(init_token='<bos>', eos_token='<eos>', batch_first=True, tokenize=char_tokenize)
        fields = [('src', text), ('trg', text)]
        examples = [Example.fromlist([m,o], fields) for m,o in zip(src, trg)]
        train_exs, val_exs = train_test_split(examples,
            test_size=self.config.val_set_size, random_state=self.config.random_seed)

        self.train_ds, self.val_ds = Dataset(train_exs, fields), Dataset(val_exs, fields)
        text.build_vocab(self.train_ds, max_size=self.config.hp.get('max_vocab_size'))

        self.text = text
        batch_size = self.config.hp.batch_size
        self.train_dataloader = data.BucketIterator(self.train_ds, batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(self.val_ds, batch_size, repeat=False, shuffle=False)

    def init_models(self):
        size = self.config.hp.model_size

        self.encoder = cudable(RNNEncoder(size, size, self.text.vocab))
        self.merge_z = cudable(FFN([size + MORPHS_SIZE, size, size]))
        self.decoder = cudable(RNNDecoder(size, size, self.text.vocab))

    def init_criterions(self):
        self.criterion = cross_entropy_without_pads(self.text.vocab)

    def init_optimizers(self):
        self.optim = Adam(chain(
            self.encoder.parameters(),
            self.merge_z.parameters(),
            self.decoder.parameters(),
        ), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Loss/train', loss.item(), self.num_iters_done)

    def loss_on_batch(self, batch):
        batch.src = cudable(batch.src)
        batch.trg = cudable(batch.trg)

        morphs = morph_chars_idx(batch.trg, self.text.vocab)
        morphs = cudable(torch.from_numpy(morphs).float())

        z = self.encoder(batch.src)
        z = self.merge_z(torch.cat([z, morphs], dim=1))

        preds = self.decoder(z, batch.trg[:, :-1])
        loss = self.criterion(preds.view(-1, len(self.text.vocab)), batch.trg[:, 1:].contiguous().view(-1))

        return loss

    def validate(self):
        losses = [self.loss_on_batch(b).item() for b in self.val_dataloader]

        self.writer.add_scalar('Loss/val', np.mean(losses), self.num_iters_done)
        self.losses['val_loss'].append(np.mean(losses))

        self.validate_inference()

    def validate_inference(self):
        generated = []
        sources = []
        gold = []

        for batch in self.val_dataloader:
            batch.src = cudable(batch.src)
            batch.trg = cudable(batch.trg)

            # Trying to reconstruct from first 3 letters
            first_chars_embs = self.decoder.embed(batch.trg[:, :4])

            morphs = morph_chars_idx(batch.trg, self.text.vocab)
            morphs = cudable(torch.from_numpy(morphs).float())

            z = self.encoder(batch.src)
            z = self.merge_z(torch.cat([z, morphs], dim=1))
            z = self.decoder.gru(first_chars_embs, z.unsqueeze(0))[1].squeeze()
            out = inference(self.decoder, z, self.text.vocab, max_len=30)

            first_chars = batch.trg[:, :4].cpu().numpy().tolist()
            results = [s + p for s,p in zip(first_chars, out)]
            results = itos_many(results, self.text.vocab, sep='')

            generated.extend(results)
            sources.extend(itos_many(batch.src, self.text.vocab, sep=''))
            gold.extend(itos_many(batch.trg, self.text.vocab, sep=''))

        # Let's try to measure BLEU scores (not valid, but at least something)
        sents_generated = [' '.join(list(s[4:])) for s in generated]
        sents_gold = [' '.join(list(s[4:])) for s in gold]
        bleu = compute_bleu_for_sents(sents_generated, sents_gold)
        self.writer.add_scalar('BLEU', bleu, self.num_iters_done)

        texts = ['{} ({}) => {}'.format(g[:3],g,p) for p,g in zip(generated, gold)]
        texts = random.sample(texts, 10) # Limiting amount of displayed text
        text = '\n\n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
