import random


def seq_noise(seq, dropout_p=0.1, shuffle_window=3):
    """Adds noise to a sequence"""
    seq = seq_dropout(seq, dropout_p)

    # TODO: can we replace it with just reversing?
    seq = gentle_shuffle(seq, shuffle_window)

    return seq


def seq_noise_many(seqs, **kwargs):
    return [seq_noise(seq, **kwargs) for seq in seqs]


def seq_dropout(seq, prob=0.1):
    return [t for t in seq if random.random() > prob]


def gentle_shuffle(seq, window_size=3):
    permutation = list(range(len(seq)))
    swap_prob = 1 / (window_size * 2 + 1)

    #print('Before:', permutation)

    for i in range(len(seq)):
        if permutation[i] != i: continue # Already swapped

        possible_swaps = range(i - window_size, i + window_size + 1)
        swap = random.choice(possible_swaps)

        if swap < 0 or swap >= len(seq): continue
        if permutation[swap] != swap: continue # We shouldn't swap a swapped element

        permutation[i], permutation[swap] = swap, i

    #print('After: ', permutation)

    return [seq[i] for i in permutation]
