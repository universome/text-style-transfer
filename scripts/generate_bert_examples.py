#!/usr/bin/env python

from copy import deepcopy
from typing import Set

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm

device = 'cuda'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-models/negative-fine-tuned-uncased.tar.gz').to(device)
# model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
model.eval()
positive_words = set([w.lower() for w in open('data/generated/words-positive.txt').read().splitlines()])
# positive_words = set([w.lower() for w in open('data/generated/words-positive-yelp.txt').read().splitlines()])


def predict(sentence_pairs:str) -> str:
    # Tokenizing separately, so later we can calculate length of a
    # (when)
    tokenized = [tokenizer.tokenize(a) + tokenizer.tokenize(b) for a, b in sentence_pairs]

    masks_idx = [[i for i,w in enumerate(s) if w.lower() in positive_words] for s in tokenized]
    # masks_idx = [[6]]

    masked_sents = deepcopy(tokenized)

    for j, mask in enumerate(masks_idx):
        for i in mask:
            masked_sents[j][i] = '[MASK]'

    # Removing those sentences which do not contains [MASK] words
    keep_sents_idx = set(i for i, s in enumerate(masks_idx) if len(s) > 0)
    sentence_pairs = [p for i, p in enumerate(sentence_pairs) if i in keep_sents_idx]
    tokenized = [s for i, s in enumerate(tokenized) if i in keep_sents_idx]
    masked_sents = [s for i, s in enumerate(masked_sents) if i in keep_sents_idx]
    masks_idx = [s for i, s in enumerate(masks_idx) if i in keep_sents_idx]

    tokens_idx = [tokenizer.convert_tokens_to_ids(s) for s in masked_sents]
    segments_ids = [[0] * len(tokenizer.tokenize(a)) + [1] * len(tokenizer.tokenize(b)) for a, b in sentence_pairs]

    # Convert inputs to PyTorch tensors
    tokens_tensors = [torch.tensor([s]).to(device) for s in tokens_idx]
    segments_tensors = [torch.tensor([s]).to(device) for s in segments_ids]

    predictions = [model(tokens, segments) for tokens, segments in tqdm(zip(tokens_tensors, segments_tensors))]
    # print('Top words:', tokenizer.convert_ids_to_tokens(predictions[0][0][6].topk(10)[1].cpu().numpy().tolist()))
    predicted_idx = [p[0][idx].argmax(dim=1).cpu().numpy().tolist() for p, idx in zip(predictions, masks_idx)]
    predicted_tokens = [tokenizer.convert_ids_to_tokens(p) for p in predicted_idx]

    results = deepcopy(masked_sents)

    for j, s in enumerate(results):
        for i, pred in zip(masks_idx[j], predicted_tokens[j]):
            s[i] = pred

    with open('bert-examples.txt', 'w') as f:
        for tok, masked, result in zip(tokenized, masked_sents, results):
            f.write('Source: ' + ' '.join(tok) + '\n')
            f.write('Masked: ' + ' '.join(masked) + '\n')
            f.write('Result: ' + ' '.join(result) + '\n')
            f.write('\n')


def main():
    docs = open('data/yelp/bert-positive.txt').read().split('\n\n')
    sentence_pairs = [tuple(d.splitlines()[:2]) for d in docs]
    sentence_pairs = [p for p in sentence_pairs if len(p) >= 2]
    sentence_pairs = sentence_pairs[:50]
    # sentence_pairs = [("What a peaceful place .", "I love going here every day .")]
    # sentence_pairs = [("Who was Jim Henson ?", "Jim Henson was a puppeteer")]
    predict(sentence_pairs)


if __name__ == '__main__':
    main()
