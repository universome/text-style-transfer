class BaseDataloader(object):
    def shuffle(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        if self._iter_count < self._n_batch:
            return self.step()
        else:
            self.reset()

    def reset(self):
        if self._should_shuffle: self.shuffle()

        self._iter_count = 0
        raise StopIteration()
