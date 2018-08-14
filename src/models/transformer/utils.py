import torch
import numpy as np
from firelab.utils import cudable


def pad_mask(seq, vocab):
    return seq != vocab.stoi['<pad>']


def subsequent_mask(size):
    "Mask out subsequent positions."

    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = cudable(torch.from_numpy(mask) == 0)

    return mask
