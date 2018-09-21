import os
import re
import pickle
import random
import sys; sys.path.extend(['.'])
from typing import List

import numpy as np
import torch
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab.utils.training_utils import cudable
from firelab.utils.fs_utils import load_config

from src.models import RNNLM
from src.utils.data_utils import itos_many, char_tokenize, word_base
from src.morph import morph_chars_idx, MORPHS_SIZE
from src.inference import InferenceState

from utils import create_get_path_fn

# Some constants
EOS_TOKEN = '|'
MAX_LINE_LEN = 256
MAX_CONTEXT_SIZE = 512
config = load_config('experiments/char-rnn/config.yml')

# models_dir = 'experiments/char-wm/checkpoints'
get_path = create_get_path_fn('dialog_models')

# Loading vocab
field = Field(eos_token=EOS_TOKEN, batch_first=True,
              tokenize=char_tokenize, pad_first=True)
field.vocab = pickle.load(open(get_path('vocab', 'pickle'), 'rb'))

print('Loading models..')
location = None if torch.cuda.is_available() else 'cpu'
lm = cudable(RNNLM(config.hp.model_size, field.vocab, n_layers=config.hp.n_layers)).eval()
lm.load_state_dict(torch.load(get_path('lm'), map_location=location))


def predict(sentences:List[str], n_lines:int, temperature:float=1e-5):
    "For each sentence generates `n_lines` lines sequentially to form a dialog"

    dialogs = [s for s in sentences] # Let's not mutate original list and copy it
    batch_size = len(dialogs)

    for _ in range(n_lines):
        examples = [Example.fromlist([EOS_TOKEN.join(d)], [('text', field)]) for d in dialogs]
        dataset = Dataset(examples, [('text', field)])
        dataloader = data.BucketIterator(dataset, batch_size, shuffle=False, repeat=False)
        batch = cudable(next(iter(dataloader))) # We have a single batch
        text = batch.text[:, -MAX_CONTEXT_SIZE:] # As we made pad_first we are not afraid of losing information
        embs = lm.embed(text)
        z = lm.gru(embs)[1]

        next_lines = InferenceState({
            'model': lm,
            'inputs': z,
            'vocab': field.vocab,
            'max_len': MAX_LINE_LEN,
            'bos_token': EOS_TOKEN, # We start infering a new reply when we see EOS
            'eos_token': EOS_TOKEN,
            'temperature': temperature,
            'sample_type': 'sample',
            'inputs_batch_first': False
        }).inference()

        next_lines = itos_many(next_lines, field.vocab, sep='')
        next_lines = [slice_unfinished_sentence(l) for l in next_lines]
        dialogs = [d + EOS_TOKEN + l for d, l in zip(dialogs, next_lines)]

    dialogs = [d.split(EOS_TOKEN) for d in dialogs]
    dialogs = [[s for s in d if len(s) != 0] for d in dialogs]
    dialogs = [assign_speakers(d) for d in dialogs]

    return dialogs


def slice_unfinished_sentence(s):
    if len(s) < MAX_LINE_LEN: return s # Line was finished in this way by itself
    if s.rfind('.') == -1: return s # We can't properly finish this line

    return s[:s.rfind('.')+1]


def assign_speakers(dialog, speakers=('Bes', 'Borgy')):
    # Generating sequence of 0,1,0,1,0,...
    turns = np.tile([0,1], len(dialog))

    # Choosing if it starts with 1 or 0
    if random.random() > 0.5:
        turns = np.concatenate([[1], turns], axis=0)

    turns = turns[:len(dialog)]

    dialog = [{'speaker': speakers[t], 'text': s} for s,t in zip(dialog, turns)]

    return dialog
