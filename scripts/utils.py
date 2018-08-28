from typing import List


def read_corpus(data_path: str) -> List[str]:
    with open(data_path) as f:
        lines = f.read().splitlines()

    return lines


def save_corpus(corpus: List[str], path: str):
    with open(path, 'w') as f:
        for line in corpus:
            f.write(line + '\n')
