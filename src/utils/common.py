import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def variable(x, **kwargs):
    v = Variable(x, **kwargs)

    return v.cuda() if use_cuda else v
