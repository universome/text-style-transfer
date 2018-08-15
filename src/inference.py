import torch
import numpy as np
from firelab.utils import cudable


def inference(model, z, vocab, enc_mask=None, max_len=100):
    """
    All decoder models have the same inference procedure
    Let's move it into the common function
    """
    batch_size = z.size(0)
    BOS, EOS = vocab.stoi['<bos>'], vocab.stoi['<eos>']
    active_seqs = cudable(torch.tensor([[BOS] for _ in range(batch_size)]).long())
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

        next_tokens = next_tokens_dists.max(dim=-1)[1][:,-1]
        active_seqs = torch.cat((active_seqs, next_tokens.unsqueeze(1)), dim=-1)
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
