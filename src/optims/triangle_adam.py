"NoamOpt. Taken from https://github.com/harvardnlp/annotated-transformer"
from torch.optim import Adam


class TriangleAdam:
    "Optim wrapper that makes Adam triangle."
    def __init__(self, parameters, config):
        self.optimizer = Adam(parameters, lr=config.adam.lr, betas=config.adam.betas)
        self._step = 0
        self.warmup = config.warmup
        self.factor = config.factor
        self.model_size = config.model_size
        self._current_lr = 0

    def step(self):
        "Update parameters and lr"
        self._step += 1
        lr = self.compute_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._current_lr = lr
        self.optimizer.step()

    def compute_lr(self, step=None):
        step = step or self._step

        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
