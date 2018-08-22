import torch as T


def embed(embedder, x, onehot=True):
    if onehot: return embedder(x)

    # TODO: zero out pad tokens
    out = T.matmul(x, embedder.weight)

    return out
