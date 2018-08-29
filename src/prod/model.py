import os
import pickle
import sys; sys.path.extend(['.'])

import numpy as np
import torch
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab.utils.training_utils import cudable

from src.models import FFN
from src.models.dissonet import RNNEncoder, RNNDecoder
from src.utils.data_utils import itos_many, char_tokenize
from src.morph import morph_chars_idx, MORPHS_SIZE
from src.inference import inference


# Some constants
batch_size = 128

# Loading vocab
text = pickle.load(open('experiments/word-filling/checkpoints/text.pickle', 'rb'))
fields = [('src', text), ('trg', text)]

# Defining model
encoder = cudable(RNNEncoder(512, 512, text.vocab))
decoder = cudable(RNNDecoder(512, 512, text.vocab))
merge_z = cudable(FFN([512 + MORPHS_SIZE, 512, 512]))

encoder.load_state_dict(torch.load('experiments/word-filling/checkpoints/encoder-112975.pth'))
decoder.load_state_dict(torch.load('experiments/word-filling/checkpoints/decoder-112975.pth'))
merge_z.load_state_dict(torch.load('experiments/word-filling/checkpoints/merge_z-112975.pth'))


def predict(sentences):
    # Splitting sentences into batches
    src, trg = generate_batch(sentences)
    examples = [Example.fromlist([m,o], fields) for m,o in zip(src, trg)]
    ds = Dataset(examples, fields)
    dataloader = data.BucketIterator(ds, batch_size, repeat=False)

    word_translations = []

    for batch in dataloader:
        # Generating predictions
        batch.src, batch.trg = cudable(batch.src), cudable(batch.trg)
        morphs = morph_chars_idx(batch.trg, text.vocab)
        morphs = cudable(torch.from_numpy(morphs).float())

        z = encoder(batch.src)
        z = merge_z(torch.cat([z, morphs], dim=1))
        first_chars_embs = decoder.embed(batch.trg[:, :4])
        z = decoder.gru(first_chars_embs, z.unsqueeze(0))[1].squeeze()
        out = inference(decoder, z, text.vocab, max_len=30)

        first_chars = batch.trg[:, :4].cpu().numpy().tolist()
        results = [s + p for s,p in zip(first_chars, out)]
        results = itos_many(results, text.vocab, sep='')

        word_translations.extend(results)

    return fill_words(sentences, word_translations)


def generate_batch(sentences):
    DROP = '__DROP__'
    CONTEXT_SIZE = 3
    src, trg = [], []

    for s in sentences:
        tokens = s.split()

        for i, t in enumerate(tokens):
            context_left = tokens[max(i - CONTEXT_SIZE, 0) : i]
            context_right = tokens[i+1 : i + CONTEXT_SIZE + 1]
            src.append(' '.join(context_left + [DROP] + context_right))
            trg.append(t)

    return src, trg


def fill_words(sentences, words):
    lens = [len(s.split()) for s in sentences]

    assert sum(lens) == len(words)

    w_idx = 0
    s_idx = 0
    results = [[] for _ in lens]

    for w in words:
        if w_idx >= lens[s_idx]:
            s_idx += 1
            w_idx = 0
            results[s_idx].append(w)
        else:
            w_idx += 1
            results[s_idx].append(w)

    return [' '.join(s) for s in results]
