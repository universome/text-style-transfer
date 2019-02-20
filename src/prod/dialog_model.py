import os
import re
import pickle
import random
import sys; sys.path.extend(['.'])
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchtext import data
from torchtext.data import Field, Dataset, Example
from firelab.utils.training_utils import cudable
from firelab.utils.fs_utils import load_config

from src.models import RNNLM, ConditionalLM
from src.utils.data_utils import itos_many, char_tokenize, word_base
from src.morph import morph_chars_idx, MORPHS_SIZE
from src.inference import InferenceState

from utils import create_get_path_fn
from throwaway_models import CharLMFromEmbs, WeightedLMEnsemble

# Some constants
EOS_TOKEN = '|'
DEFAULT_MAX_LINE_LEN = 256
MAX_CONTEXT_SIZE = 512
DEFAULT_TEMPERATURE = 0.25
MODEL_CLASSES = {
    'RNNLM': RNNLM,
    'ConditionalLM': ConditionalLM,
    'CharLMFromEmbs': CharLMFromEmbs,
}

def init_lm(config_path, state_path, model_cls_name:str):
    model_cls = MODEL_CLASSES[model_cls_name]
    hp = load_config(config_path).get('hp')
    get_path = create_get_path_fn(state_path)

    # Loading vocab
    field = Field(eos_token=EOS_TOKEN, batch_first=True,
                  tokenize=char_tokenize, pad_first=True)
    field.vocab = pickle.load(open(get_path('vocab', 'pickle'), 'rb'))

    print('Loading models..')
    device = None if torch.cuda.is_available() else 'cpu'

    if model_cls is RNNLM:
        lm = cudable(RNNLM(hp.model_size, field.vocab, n_layers=hp.n_layers)).eval()
        lm.load_state_dict(torch.load(get_path('lm'), map_location=device))
    elif model_cls is ConditionalLM:
        lm = cudable(ConditionalLM(hp.model_size, field.vocab)).eval()
        lm.load_state_dict(torch.load(get_path('lm'), map_location=device))
    elif model_cls is CharLMFromEmbs:
        rnn_lm = cudable(RNNLM(hp.model_size, field.vocab, n_layers=hp.n_layers))
        style_embed = cudable(nn.Embedding(2, hp.model_size))

        rnn_lm.load_state_dict(torch.load(get_path('lm'), map_location=device))
        style_embed.load_state_dict(torch.load(get_path('style_embed'), map_location=device))

        lm = cudable(CharLMFromEmbs(rnn_lm, style_embed, n_layers=hp.n_layers)).eval()
    else:
        raise NotImplementedError

    return lm, field


def build_predict_fn(model_dir:str, model_cls_name:str, inference_kwargs={}, ensemble_models=None, ensemble_weights=None):
    if model_cls_name != 'WeightedLMEnsemble':
        lm, field = init_lm('%s/config.yml' % model_dir, '%s/state' % model_dir, model_cls_name)
    else:
        lms = []
        field = None

        for sub_model_dir, sub_model_cls_name in ensemble_models:
            lm, field = init_lm('%s/config.yml' % sub_model_dir, '%s/state' % sub_model_dir, sub_model_cls_name)
            lms.append(lm)

        lm = WeightedLMEnsemble(lms, ensemble_weights)

    def predict(sentences:List[str], n_lines:int, temperature:float=None, max_len:int=None):
        "For each sentence generates `n_lines` lines sequentially to form a dialog"
        dialogs = [s for s in sentences] # Let's not mutate original list and copy it
        batch_size = len(dialogs)
        temperature = temperature or DEFAULT_TEMPERATURE
        max_len = max_len or DEFAULT_MAX_LINE_LEN

        for _ in range(n_lines):
            examples = [Example.fromlist([EOS_TOKEN.join(d)], [('text', field)]) for d in dialogs]
            dataset = Dataset(examples, [('text', field)])
            dataloader = data.BucketIterator(dataset, batch_size, shuffle=False, repeat=False)
            batch = next(iter(dataloader)) # We have a single batch
            text = cudable(batch.text[:, -MAX_CONTEXT_SIZE:]) # As we made pad_first we are not afraid of losing information

            if model_cls_name == 'CharLMFromEmbs':
                z = lm.init_z(text.size(0), 1)
                z = lm(z, text, return_z=True)[1]
            elif model_cls_name == 'ConditionalLM':
                z = cudable(torch.zeros(2, len(text), 2048))
                z = lm(z, text, style=1, return_z=True)[1]
            elif model_cls_name == 'WeightedLMEnsemble':
                z = cudable(torch.zeros(2, 1, len(text), 4096))
                z = lm(z, text, return_z=True)[1]
            else:
                embs = lm.embed(text)
                z = lm.gru(embs)[1]

            next_lines = InferenceState({
                'model': lm,
                'inputs': z,
                'vocab': field.vocab,
                'max_len': max_len,
                'bos_token': EOS_TOKEN, # We start infering a new reply when we see EOS
                'eos_token': EOS_TOKEN,
                'temperature': temperature,
                'sample_type': 'sample',
                'inputs_batch_dim': 1 if model_cls_name != 'WeightedLMEnsemble' else 2,
                'substitute_inputs': True,
                'kwargs': inference_kwargs
            }).inference()

            next_lines = itos_many(next_lines, field.vocab, sep='')
            next_lines = [slice_unfinished_sentence(l) for l in next_lines]
            dialogs = [d + EOS_TOKEN + l for d, l in zip(dialogs, next_lines)]

        dialogs = [d.split(EOS_TOKEN) for d in dialogs]
        dialogs = [[s for s in d if len(s) != 0] for d in dialogs]
        dialogs = [assign_speakers(d) for d in dialogs]

        return dialogs

    return predict


def slice_unfinished_sentence(s):
    if len(s) < DEFAULT_MAX_LINE_LEN: return s # Line was finished in this way by itself
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


# Default
# classic_lm = init_lm('experiments/char-rnn/config.yml', 'dialog_models')
# subs_lm, subs_field = init_lm('subs_lm/config.yml', 'subs_lm/state')
# classic_lm, classic_field = init_lm('classic_lm/config.yml', 'classic_lm/state')
print('Loading default dialog model..')
# predict = build_predict_fn('conditional_lm', 'ConditionalLM', {'style': 1})
# predict = build_predict_fn('fine_tuned_classic_lm', 'RNNLM')
predict = build_predict_fn('models/classic-lm', 'RNNLM', inference_kwargs={'return_z': True})
# predict = build_predict_fn(None, 'WeightedLMEnsemble',
#     ensemble_models=[('RNNLM', 'classic_lm'), ('RNNLM', 'overfitted_fine_tuned_classic_lm')],
#     ensemble_weights=[0.5, 0.5]
# )
