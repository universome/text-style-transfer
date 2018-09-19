import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from firelab.utils.training_utils import cudable
from torch.distributions import Categorical

from src.utils.gumbel import gumbel_softmax_sample


class InferenceState:
    def __init__(self, state_dict:dict):
        # Required arguments
        self.model = state_dict['model']
        self.vocab = state_dict['vocab']
        self.inputs = state_dict['inputs']

        # Optional arguments
        self.bos_token = state_dict.get('bos_token', '<bos>')
        self.eos_token = state_dict.get('eos_token', '<eos>')
        self.pad_token = state_dict.get('pad_token', '<pad>')
        self.unk_token = state_dict.get('unk_token', '<unk>')
        self.bos_idx = self.vocab.stoi[self.bos_token]
        self.eos_idx = self.vocab.stoi[self.eos_token]
        self.pad_idx = self.vocab.stoi[self.pad_token]
        self.unk_idx = self.vocab.stoi[self.unk_token]
        self.max_len = state_dict.get('max_len', 100)
        self.temperature = state_dict.get('temperature', 1)
        self.sample_type = state_dict.get('sample_type', 'max') # max or sample
        self.kwargs = state_dict.get('kwargs', {})
        self.gumbel = state_dict.get('gumbel', False) # TODO: decompose into add_gumbel_noise/ste/differentiable args
        self.enc_mask = state_dict.get('enc_mask')
        self.should_stack_finished = state_dict.get('should_stack_finished', False)
        self.inputs_batch_first = state_dict.get('inputs_batch_first', True)
        self.batch_size = self.inputs.size(0 if self.inputs_batch_first else 1)
        self.active_seqs = state_dict.get('active_seqs', self.generate_active_seqs())

        # Inner properties
        self.finished = [None for _ in range(self.batch_size)]
        self.active_seqs_idx = T.arange(self.batch_size)
        self.should_stop = False
        self.num_steps_done = 0

        self.validate()

    def generate_active_seqs(self):
        return cudable(T.tensor([[self.bos_idx] for _ in range(self.batch_size)]).long())

    def validate(self):
        assert self.max_len > 0
        assert self.batch_size > 0
        assert type(self.bos_token) is str
        assert type(self.eos_token) is str

        # Checking that our bos/eos token idx are in the dictionary
        assert self.bos_idx != self.vocab.stoi[self.unk_token]
        assert self.eos_idx != self.vocab.stoi[self.unk_token]

    def is_finished(self):
        num_active = sum([s is None for s in self.finished])

        return num_active == 0 or self.num_steps_done >= self.max_len

    def force_finish(self):
        "Finishes all the sentences because max length was exceeded"
        # TODO(universome): finished[active_idx] = active
        for i, seq in zip(self.active_seqs_idx, self.active_seqs):
            self.finished[i] = seq

    def stack_finished(self):
        "Pads finished sequences with <pad> token and stacks into tensor"
        max_len = max(len(s) for s in self.finished)

        for i, seq in enumerate(self.finished):
            self.finished[i] = self.pad_to_max(seq, max_len)

        self.finished = cudable(T.stack(self.finished))

    def pad_to_max(self, seq, max_len):
        if len(seq) == max_len: return seq

        if self.gumbel:
            # TODO: we fill with eps to prevent numerical issues in loss computation later
            # But loss on pads is always zero, why are we doing this?
            eps = 1e-8
            pads = cudable(T.zeros(max_len - len(seq), len(self.vocab)))
            pads = pads.fill_(eps).index_fill_(1, T.tensor(self.pad_idx), 1.)
        else:
            pads = cudable(T.zeros(max_len - len(seq)).fill_(self.pad_idx))

        return T.cat((seq, pads), dim=0)

    def update_finished(self):
        "Adding finished sequences to the `finished` list"
        just_finished_idx = self.active_seqs_idx[self.finished_mask()]

        # TODO: finished[just_finished_idx] = active.masked_select(next_tokens == EOS)
        for i, seq in zip(just_finished_idx, self.active_seqs[self.finished_mask()]):
            self.finished[i] = seq

    def update_active_seqs(self):
        "Removing finished sequences from the batch"
        next_x = self.next_tokens if not self.gumbel else self.next_tokens_dists
        self.active_seqs = T.cat([self.active_seqs, next_x.unsqueeze(1)], dim=-1)
        self.active_seqs = self.active_seqs[~self.finished_mask()]
        self.active_seqs_idx = self.active_seqs_idx[~self.finished_mask()]
        self.inputs = self.inputs[~self.finished_mask()] if self.inputs_batch_first else self.inputs[:, ~self.finished_mask()]

        if not self.enc_mask is None:
            self.enc_mask = self.enc_mask[~self.finished_mask()]

    def forward(self):
        # TODO: let's hard-code single encoder mask arg until we'll need something more general
        args = [] if self.enc_mask is None else self.enc_mask
        next_tokens_dists = self.model.forward(self.inputs, self.active_seqs, *args, **self.kwargs)[:, -1]
        next_tokens = self.sample(next_tokens_dists)

        if self.gumbel:
            next_tokens_dists = F.softmax(next_tokens_dists, dim=1)
            next_tokens_dists = gumbel_softmax_sample(next_tokens_dists, self.temperature)

        self.next_tokens = next_tokens
        self.next_tokens_dists = next_tokens_dists

    def sample(self, token_dists):
        if self.sample_type == 'max':
            return token_dists.max(dim=-1)[1]
        elif self.sample_type == 'sample':
            return Categorical(logits=token_dists/self.temperature).sample()
        else:
            raise NotImplementedError

    def finished_mask(self):
        return self.next_tokens == self.eos_idx

    def inference(self):
        for _ in range(self.max_len):
            self.forward()
            self.update_finished()
            self.update_active_seqs()

            if self.is_finished():
                break

        self.force_finish()
        if self.should_stack_finished: self.stack_finished()

        return self.finished


def simple_inference(model, z, vocab, max_len=100):
    infered = InferenceState({
        'model': model,
        'inputs': z,
        'vocab': vocab,
        'max_len': max_len
    }).inference()

    infered = [x.cpu().numpy().tolist() for x in infered]

    return infered
