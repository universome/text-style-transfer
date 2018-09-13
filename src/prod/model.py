import os
import re
import pickle
import sys; sys.path.extend(['.'])
from typing import List

from nltk import word_tokenize
import numpy as np
import torch
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab.utils.training_utils import cudable

from src.models import FFN, RNNEncoder, RNNDecoder
from src.utils.data_utils import itos_many, char_tokenize, word_base
from src.morph import morph_chars_idx, MORPHS_SIZE
from src.inference import simple_inference


# Some constants
batch_size = 128
n_first_chars = 3

# models_dir = 'experiments/char-wm/checkpoints'
models_dir = 'models'
versions = set([int(f.split('-')[1].split('.')[0]) for f in os.listdir(models_dir) if '-' in f])
latest_iter = max(versions)
get_path = lambda m, ext='pth': os.path.join(models_dir, '{}-{}.{}'.format(m, latest_iter, ext))
print('Latest iter (version) found: {}. Loading from it.'.format(latest_iter))

# Loading vocab
field = Field(init_token='<bos>', eos_token='<eos>',
              batch_first=True, tokenize=char_tokenize)
field.vocab = pickle.load(open(get_path('vocab', 'pickle'), 'rb'))
fields = [('src', field), ('trg', field)]

print('Loading models..')
encoder = cudable(RNNEncoder(512, 512, field.vocab)).eval()
decoder = cudable(RNNDecoder(512, 512, field.vocab)).eval()
merge_z = cudable(FFN([512 + MORPHS_SIZE, 512])).eval()

location = None if torch.cuda.is_available() else 'cpu'
encoder.load_state_dict(torch.load(get_path('encoder'), map_location=location))
decoder.load_state_dict(torch.load(get_path('decoder'), map_location=location))
merge_z.load_state_dict(torch.load(get_path('merge_z'), map_location=location))


def predict(lines):
    lines = tokenize(lines)
    # Grouping all lines into batches
    src, trg = generate_dataset_with_middle_chars(lines)
    examples = [Example.fromlist([m,o], fields) for m,o in zip(src, trg)]
    ds = Dataset(examples, fields)
    dataloader = data.BucketIterator(ds, batch_size, repeat=False, shuffle=False)

    word_translations = []

    for batch in dataloader:
        # Generating predictions
        batch = cudable(batch)
        morphs = morph_chars_idx(batch.trg, field.vocab)
        morphs = cudable(torch.from_numpy(morphs).float())
        first_chars_embs = decoder.embed(batch.trg[:, :n_first_chars])

        z = encoder(batch.src)
        z = merge_z(torch.cat([z, morphs], dim=1))
        z = decoder.gru(first_chars_embs, z.unsqueeze(0))[1].squeeze()
        out = simple_inference(decoder, z, field.vocab, max_len=30)

        first_chars = batch.trg[:, :n_first_chars].cpu().numpy().tolist()
        results = [s + p for s,p in zip(first_chars, out)]
        results = itos_many(results, field.vocab, sep='')

        word_translations.extend(results)

    transfered = group_by_lens(word_translations, [len(s.split()) for s in lines])
    transfered = [mix_transfered(o,t) for o,t in zip(lines, transfered)]

    return transfered


def generate_dataset_with_contexts(sentences, context_size=3):
    DROP = '__DROP__'
    src, trg = [], []

    for s in sentences:
        tokens = s.split()

        for i, t in enumerate(tokens):
            context_left = tokens[max(i - context_size, 0) : i]
            context_right = tokens[i+1 : i + context_size + 1]
            src.append(' '.join(context_left + [DROP] + context_right))
            trg.append(t)

    return src, trg


def generate_dataset_with_middle_chars(sentences, n_chars=4):
    src = [word_base(t, n_chars) for s in sentences for t in s.split()]
    trg = [t for s in sentences for t in s.split()]

    return src, trg


def fill_words(sentences, words):
    lens = [len(s.split()) for s in sentences]
    original = [w for s in sentences for w in s.split()]

    assert sum(lens) == len(words) == len(original)

    words = [choose_word(w,o) for w,o in zip(words, original)]

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


def group_by_lens(tokens: List[str], lens: List[int]) -> List[str]:
    result = []
    curr_lens_sum = 0

    for l in lens:
        line = ' '.join(tokens[curr_lens_sum:curr_lens_sum + l])
        result.append(line)
        curr_lens_sum += l

    return result


def mix_transfered(original, transfered):
    original_tokens = original.split()
    transfered_tokens = transfered.split()

    assert len(original_tokens) == len(transfered_tokens)

    return ' '.join(choose_word(o,t) for o,t in zip(original_tokens, transfered_tokens))


def choose_word(original, translated):
    if len(original) < 5:
        return original # We do not translate short words

    if not re.match(r"^([а-я]|ё)+$", original):
        return original # Do not translate non-simple words

    if not re.match(r"^([а-я]|ё)+$", translated):
        return original # We have translated word into garbage :|

    return translated


def tokenize(sentences):
    return [' '.join(word_tokenize(s)) for s in sentences]
