#!/usr/bin/env python

from copy import deepcopy
from typing import Set

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm

device = 'cuda'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-models/pytorch_model.tar.gz').to(device)
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
model.eval()
positive_words = set([w.lower() for w in open('data/generated/words-positive.txt').read().splitlines()])


def predict(sentences:str) -> str:
    tokenized = [tokenizer.tokenize(s) for s in sentences]
    # masks_idx = [[i for i,w in enumerate(s) if w.lower() in positive_words] for s in tokenized]
    masks_idx = [[6]]

    masked_sents = deepcopy(tokenized)
    for j, mask in enumerate(masks_idx):
        for i in mask:
            masked_sents[j][i] = '[MASK]'

    # Removing those sentences which do not contains [MASK] words
    tokenized = [s for i, s in enumerate(tokenized) if len(masks_idx[i]) > 0]
    masked_sents = [s for i, s in enumerate(masked_sents) if len(masks_idx[i]) > 0]
    masks_idx = [s for i, s in enumerate(masks_idx) if len(masks_idx[i]) > 0]

    tokens_idx = [tokenizer.convert_tokens_to_ids(s) for s in masked_sents]
    segments_ids = [[0] * len(s) for s in tokens_idx]
    segments_ids = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]

    # Convert inputs to PyTorch tensors
    tokens_tensors = [torch.tensor([s]).to(device) for s in tokens_idx]
    segments_tensors = [torch.tensor([s]).to(device) for s in segments_ids]

    preds = [model(tokens, segments) for tokens, segments in tqdm(zip(tokens_tensors, segments_tensors))]
    predictions = [model(tokens, segments) for tokens, segments in tqdm(zip(tokens_tensors, segments_tensors))]
    predicted_idx = [p[0][idx].argmax(dim=1).cpu().numpy().tolist() for p, idx in zip(predictions, masks_idx)]
    predicted_tokens = [tokenizer.convert_ids_to_tokens(p) for p in predicted_idx]

    results = deepcopy(masked_sents)

    for j, s in enumerate(results):
        for i, pred in zip(masks_idx[j], predicted_tokens[j]):
            s[i] = pred

    for tok, masked, result in zip(tokenized, masked_sents, results):
        print('Source:', ' '.join(tok))
        print('Masked:', ' '.join(masked) )
        print('Result:', ' '.join(result))
        print()


def main():
    # sentences = open('data/yelp/yelp-positive-short.en').read().splitlines()
    # sentences = list(filter(len, sentences))
    # sentences = sentences[:10]
    sentences = ['Who was Jim Henson ? Jim Henson was a puppeteer']
    predict(sentences)


if __name__ == '__main__':
    main()
