import torch
from torch.autograd import Variable

import src.transformer.constants as constants


use_cuda = torch.cuda.is_available()

def variable(x, **kwargs):
    v = Variable(x, **kwargs)

    return v.cuda() if use_cuda else v


def one_hot_to_ints(one_hots):
    '''
    Converts batch of 1-hot encoded vectors into integers
    We do it by multiplying each vector with vector [0,1,2,3,4,...,n]
    '''
    ints = torch.arange(one_hots.size(1))
    if use_cuda: ints = ints.cuda()
    if type(one_hots) == Variable: ints = Variable(ints, requires_grad=False)

    return torch.matmul(one_hots, ints).long()


def embed(x, embeddings):
    if not src_seq.requires_grad: return embeddings(src_seq)

    # Applying embeddings manually :|
    out = tf.matmul(src_seq, embeddings.weight)
    shape = out.size()
    pad_idx = src_seq == constants.PAD
    out = out.view(-1, shape[2]).transpose(0,1).masked_fill_(pad_idx.view(-1), 0)
    out = out.transpose(0,1).view(*shape)

    return out

