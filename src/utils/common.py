import torch
from torch.autograd import Variable

import src.transformer.constants as constants

use_cuda = torch.cuda.is_available()


def variable(x, **kwargs):
    v = Variable(x, **kwargs)

    return v.cuda() if use_cuda else v


def embed(x, embeddings, one_hot_input=False):
    if not one_hot_input: return embeddings(x)

    # Applying embeddings manually :|
    # The problem here is that we need to zero out PAD symbol
    out = torch.matmul(x, embeddings.weight)
    shape = out.size()
    pad_idx = one_hot_seqs_to_seqs(x) == constants.PAD
    if type(out) is Variable: pad_idx = Variable(pad_idx, requires_grad=False)
    out = out.view(-1, shape[2]).transpose(0,1)
    out.masked_fill_(pad_idx.view(-1), 0)
    out = out.transpose(0,1).view(*shape)

    return out


def one_hot_seqs_to_seqs(seqs):
    assert seqs.dim() == 3

    # TODO: does it really fast than torch.max(x, 2)[1] ?

    if type(seqs) is Variable: seqs = seqs.data

    indices = torch.arange(seqs.size(2))
    if use_cuda: indices = indices.cuda()

    return torch.matmul(seqs.float(), indices).long()