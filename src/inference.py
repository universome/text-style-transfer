import torch as T
import numpy as np
from firelab.utils import cudable

from src.utils.gumbel import gumbel_softmax_sample


def inference(model, z, vocab, enc_mask=None, max_len=100):
    "Common inference procedure for different decoders"
    batch_size = z.size(0)
    BOS, EOS = vocab.stoi['<bos>'], vocab.stoi['<eos>']
    active_seqs = cudable(T.tensor([[BOS] for _ in range(batch_size)]).long())
    active_seqs_idx = np.arange(batch_size)
    finished = [None for _ in range(batch_size)]
    n_finished = 0

    for _ in range(max_len):
        # TODO: use beam search
        # TODO: this code looks awful.
        if enc_mask is None:
            next_tokens_dists = model.forward(z, active_seqs)
        else:
            next_tokens_dists = model.forward(z, active_seqs, enc_mask)

        # TODO: first do [:,-1], then .max(...), cause it's faster this way
        next_tokens = next_tokens_dists.max(dim=-1)[1][:,-1]
        active_seqs = T.cat((active_seqs, next_tokens.unsqueeze(1)), dim=-1)
        finished_mask = (next_tokens == EOS).cpu().numpy().astype(bool)
        finished_seqs_idx = active_seqs_idx[finished_mask]
        active_seqs_idx = active_seqs_idx[finished_mask == 0]
        n_finished += finished_seqs_idx.size

        if finished_seqs_idx.size != 0:
            # TODO(universome)
            # finished[finished_seqs_idx] = active_seqs.masked_select(next_tokens == EOS).cpu().numpy()
            for i, seq in zip(finished_seqs_idx, active_seqs[next_tokens == EOS]):
                finished[i] = seq.cpu().numpy().tolist()

            active_seqs = active_seqs[next_tokens != EOS]
            z = z[next_tokens != EOS]
            if not enc_mask is None:
                enc_mask = enc_mask[next_tokens != EOS]

        if n_finished == batch_size: break

    # Well, some sentences were finished at the time
    # Let's just fill them in
    if n_finished != batch_size:
        # TODO(universome): finished[active_seqs_idx] = active_seqs
        for i, seq in zip(active_seqs_idx, active_seqs):
            finished[i] = seq.cpu().numpy().tolist()

    return finished


def gumbel_inference(model, memory, vocab, t=1, max_len=100):
    "Differentiable inference with Gumbel softmax"
    batch_size = memory.size(0)
    BOS, EOS, PAD = vocab.stoi['<bos>'], vocab.stoi['<eos>'], vocab.stoi['<pad>']
    active = cudable(T.zeros(batch_size, 1, len(vocab)).index_fill_(2, T.tensor(BOS), 1.))
    active_idx = np.arange(batch_size)
    finished = [None for _ in range(batch_size)]
    n_finished = 0

    for _ in range(max_len):
        next_tokens_dists = model.forward(memory, active, None, onehot=False)[:,-1]
        next_tokens_dists = gumbel_softmax_sample(next_tokens_dists, t)
        finished_mask = next_tokens_dists.max(dim=-1)[1] == EOS
        finished_mask_np = finished_mask.cpu().numpy().astype(bool)
        active = T.cat((active, next_tokens_dists.unsqueeze(1)), dim=1)
        # finished_mask = (next_tokens == EOS).cpu().numpy().astype(bool)
        finished_seqs_idx = active_idx[finished_mask_np]
        active_idx = active_idx[finished_mask_np == 0]
        n_finished += finished_seqs_idx.size

        if finished_seqs_idx.size != 0:
            # TODO(universome)
            # finished[finished_seqs_idx] = active.masked_select(next_tokens == EOS).cpu().numpy()
            for i, seq in zip(finished_seqs_idx, active[finished_mask]):
                finished[i] = seq

            active = active[~finished_mask]
            memory = memory[~finished_mask]

        if n_finished == batch_size: break

    # Well, some sentences were finished at the time. Let's just fill them in.
    if n_finished != batch_size:
        # TODO(universome): finished[active_idx] = active
        for i, seq in zip(active_idx, active):
            finished[i] = seq

    # Now let's fill short sentences with pads
    max_len = max(len(s) for s in finished)
    for i, seq in enumerate(finished):
        assert len(seq) <= max_len and seq.dim() == 2
        if len(seq) != max_len:
            pads = cudable(T.zeros(max_len - len(seq), len(vocab)).index_fill_(2, T.tensor(PAD), 1.))
            finished[i] = T.cat((seq, pads), dim=0)

    finished = cudable(T.stack(finished))

    return finished
