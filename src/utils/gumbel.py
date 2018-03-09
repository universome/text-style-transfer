'''
Helpful functions to work with Gumbel distribution
Taken from https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
'''

import torch
import torch.nn.functional as F
from src.utils.common imoprt variable


use_cuda = torch.cuda.is_available()


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    G = variable(-torch.log(-torch.log(U + eps) + eps), requires_grad=False)

    return G


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())

    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] a one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, idx = y.max(dim=-1)

    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, idx.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    if use_cuda: y_hard = y_hard.cuda()

    return (y_hard - y).detach() + y
