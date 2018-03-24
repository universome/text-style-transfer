import torch
import torch.nn as nn


use_cuda = torch.cuda.is_available()


def cross_entropy_without_pads(vocab_size):
    ''' With PAD token zero weight '''
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0

    if use_cuda: weight = weight.cuda()

    return nn.CrossEntropyLoss(weight)
